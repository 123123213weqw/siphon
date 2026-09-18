"""RWKV7 (G1j) forward pass for Siphon.

A hand-written forward pass (no transformers, no FLA) that loads an
HF-format RWKV7 checkpoint with the Siphon loader and runs it with fp16
weights / fp32 Wkv state. The per-layer structure follows the FLA reference
implementation and the llama.cpp graph builder:

    h = attn_norm(residual)
    delta = h_prev - h                      (token shift, 0 -> -h for new seqs)
    xr..xg = h + delta * x_*
    r = r_proj(xr); k = k_proj(xk); v = v_proj(xv)
    w = w_scale * sigmoid(w_lora(xw))       (kernel applies exp())
    v = lerp(v, v_first, sigmoid(v_lora(xv)))      (layers >= 1)
    a = sigmoid(a_lora(xa)); g = g_l1(sigmoid(g_l0(xg)))
    kk = l2norm(k * k_k); k = k * (1 + (a - 1) * k_a)
    o = wkv(r, w, k, v, a=-kk, b=kk*a, state)      (per-head scalar, fp32)
    o = group_norm(o); o = (o + rk * v) * g        rk = (r*k*r_k).sum(-1) per head
    x = residual + o_proj(o)

    h = ffn_norm(residual); delta = h_prev - h
    x = residual + value(sqrelu(key(h + delta * x_k)))
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RWKV7Config
from .wkv import wkv7, gemv16, wkv7_post, lora_h1, lora_gates

# Fused decode epilogue + LoRA kernels (off with SIPHON_FUSED=0).
_FUSED = os.environ.get("SIPHON_FUSED", "1") != "0"
# LoRA gate launch geometry (tuned on V100).
_LORA_PHASES = int(os.environ.get("SIPHON_LORA_PHASES", "128"))
_LORA_SPLITR = int(os.environ.get("SIPHON_LORA_SPLITR", "16"))


def sqrelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.relu(x)


@dataclass
class RWKV7State:
    """Recurrent state for continuation decoding.

    wkv:       [L, H*D] fp32
    attn_prev: [L, C] fp16 (previous normalized input; zero for a new
               sequence, which reproduces delta = -h for the first token)
    ffn_prev:  [L, C] fp16
    """
    wkv: torch.Tensor
    attn_prev: torch.Tensor
    ffn_prev: torch.Tensor


@dataclass
class _LayerParams:
    attn_norm_w: torch.Tensor
    attn_norm_b: torch.Tensor
    x: torch.Tensor                    # [6, C]  (x_r, x_w, x_k, x_v, x_a, x_g)
    r_w: torch.Tensor                  # [C, C]  (out, in) for the 3 projections
    k_w: torch.Tensor                  # [C, C]
    v_w: torch.Tensor                  # [C, C]
    o_w: torch.Tensor                  # [C, C]
    w_l0: torch.Tensor                 # [rank, C]
    w_l1: torch.Tensor                 # [rank, C]
    w_lb: torch.Tensor                 # [C]
    a_l0: torch.Tensor
    a_l1: torch.Tensor                 # [rank, C]
    a_lb: torch.Tensor
    v_l0: torch.Tensor | None          # None at layer 0
    v_l1: torch.Tensor | None
    v_lb: torch.Tensor | None
    g_l0: torch.Tensor                 # [C, rank_g]
    g_l1: torch.Tensor                 # [rank_g, C]
    k_k: torch.Tensor                  # [C]
    k_a: torch.Tensor                  # [C]
    r_k: torch.Tensor                  # [C] (head-major: [H, D] flattened)
    gn_w: torch.Tensor                 # [C]
    gn_b: torch.Tensor                 # [C]
    ffn_norm_w: torch.Tensor
    ffn_norm_b: torch.Tensor
    ffn_x_k: torch.Tensor              # [C]
    ffn_key: torch.Tensor              # [I, C]
    ffn_value: torch.Tensor            # [C, I]


@dataclass
class RWKV7Model:
    config: RWKV7Config
    device: str
    embed: torch.Tensor = field(repr=False, default=None)    # [V, C] fp16
    pre_norm_w: torch.Tensor = field(repr=False, default=None)
    pre_norm_b: torch.Tensor = field(repr=False, default=None)
    final_norm_w: torch.Tensor = field(repr=False, default=None)
    final_norm_b: torch.Tensor = field(repr=False, default=None)
    lm_head: torch.Tensor = field(repr=False, default=None)  # [C, V] fp16
    layers: list[_LayerParams] = field(repr=False, default_factory=list)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_hf_dir(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        loader: str = "siphon",
    ) -> "RWKV7Model":
        """Load an HF-format RWKV7 dir (config.json + *.safetensors).

        ``loader``: "siphon" uses the Siphon safe_open loader; "safetensors"
        uses the reference loader (for cross-checks).
        """
        model_dir = Path(model_dir)
        import json
        cfg_dict = json.loads((model_dir / "config.json").read_text())
        cfg = RWKV7Config.from_dict(cfg_dict)

        W: dict[str, torch.Tensor] = {}
        if loader == "siphon":
            from siphon import safe_open
            files = [str(f) for f in sorted(model_dir.glob("*.safetensors"))]
            assert files, f"no safetensors in {model_dir}"
            with safe_open(files, framework="pt", device=device) as f:
                for name, t in f.tensors():
                    W[name] = t
        elif loader == "safetensors":
            from safetensors.torch import safe_open as st_open
            for fname in sorted(model_dir.glob("*.safetensors")):
                with st_open(str(fname), framework="pt", device=device) as f:
                    for name in f.keys():
                        W[name] = f.get_tensor(name)
        else:
            raise ValueError(loader)

        L = cfg.num_layers
        C = cfg.hidden_size

        def get(name: str) -> torch.Tensor:
            return W[name].half().contiguous()

        def getvec(name: str) -> torch.Tensor:
            return W[name].reshape(-1).half().contiguous()

        model = cls(config=cfg, device=device)
        model.embed = get("model.embeddings.weight")
        model.pre_norm_w = getvec("model.layers.0.pre_norm.weight")
        model.pre_norm_b = getvec("model.layers.0.pre_norm.bias")
        model.final_norm_w = getvec("model.norm.weight")
        model.final_norm_b = getvec("model.norm.bias")
        model.lm_head = get("lm_head.weight").t().contiguous()

        for i in range(L):
            p = f"model.layers.{i}."
            x = torch.stack([
                getvec(p + "attn.x_r"), getvec(p + "attn.x_w"),
                getvec(p + "attn.x_k"), getvec(p + "attn.x_v"),
                getvec(p + "attn.x_a"), getvec(p + "attn.x_g"),
            ])
            if i == 0:
                v_l0 = v_l1 = v_lb = None
            else:
                v_l0 = get(p + "attn.v_lora.lora.0.weight").t().contiguous()
                v_l1 = get(p + "attn.v_lora.lora.2.weight").t().contiguous()
                v_lb = getvec(p + "attn.v_lora.lora.2.bias")
            model.layers.append(_LayerParams(
                attn_norm_w=getvec(p + "attn_norm.weight"),
                attn_norm_b=getvec(p + "attn_norm.bias"),
                x=x,
                r_w=get(p + "attn.r_proj.weight").t().contiguous(),
                k_w=get(p + "attn.k_proj.weight").t().contiguous(),
                v_w=get(p + "attn.v_proj.weight").t().contiguous(),
                o_w=get(p + "attn.o_proj.weight").t().contiguous(),
                w_l0=get(p + "attn.w_lora.lora.0.weight").t().contiguous(),
                w_l1=get(p + "attn.w_lora.lora.2.weight").t().contiguous(),
                w_lb=getvec(p + "attn.w_lora.lora.2.bias"),
                a_l0=get(p + "attn.a_lora.lora.0.weight").t().contiguous(),
                a_l1=get(p + "attn.a_lora.lora.2.weight").t().contiguous(),
                a_lb=getvec(p + "attn.a_lora.lora.2.bias"),
                v_l0=v_l0, v_l1=v_l1, v_lb=v_lb,
                g_l0=get(p + "attn.g_lora.lora.0.weight").t().contiguous(),
                g_l1=get(p + "attn.g_lora.lora.2.weight").t().contiguous(),
                k_k=getvec(p + "attn.k_k"),
                k_a=getvec(p + "attn.k_a"),
                r_k=getvec(p + "attn.r_k"),
                gn_w=getvec(p + "attn.g_norm.weight"),
                gn_b=getvec(p + "attn.g_norm.bias"),
                ffn_norm_w=getvec(p + "ffn_norm.weight"),
                ffn_norm_b=getvec(p + "ffn_norm.bias"),
                ffn_x_k=getvec(p + "ffn.x_k"),
                ffn_key=get(p + "ffn.key.weight").t().contiguous(),
                ffn_value=get(p + "ffn.value.weight").t().contiguous(),
            ))
        return model

    # ----------------------------------------------------------------- state
    def init_state(self) -> RWKV7State:
        dev = self.device
        L, C = self.config.num_layers, self.config.hidden_size
        return RWKV7State(
            wkv=torch.zeros(L, self.config.state_numel, dtype=torch.float32, device=dev),
            attn_prev=torch.zeros(L, C, dtype=torch.float16, device=dev),
            ffn_prev=torch.zeros(L, C, dtype=torch.float16, device=dev),
        )

    # --------------------------------------------------------------- forward
    def _mm(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """x [B,T,K] fp16 @ W [K,N] fp16 -> [B,T,N] fp16.

        Single-token decode (B==T==1) routes the K==hidden, N>=hidden GEMVs
        (r/k/v/o projections and ffn_key) to the custom split-K GEMV kernel,
        which beats cuBLAS on V100 for those shapes. ffn_value (K=inter,
        N=hidden) and the small LoRA GEMVs stay on cuBLAS, which wins there.
        All prefill uses cuBLAS.
        """
        C = self.config.hidden_size
        if x.shape[0] == 1 and x.shape[1] == 1 \
                and not os.environ.get("SIPHON_GEMV_OFF") \
                and W.shape[0] == C and W.shape[1] >= C:
            return gemv16(W, x.view(-1)).view(1, 1, -1)
        return torch.matmul(x, W)

    def forward(self, input_ids: torch.Tensor, state: RWKV7State,
                capture: list | None = None,
                state_out: RWKV7State | None = None) -> tuple[torch.Tensor, RWKV7State]:
        """input_ids: [B, T] int64 -> (logits [B, T, V] fp32, new_state).

        If ``state_out`` is given the new state is written into its tensors
        (in-place when ``state_out is state``); useful for CUDA-graph decode.
        """
        B, T = input_ids.shape
        cfg = self.config
        C = cfg.hidden_size
        H, D = cfg.num_heads, cfg.head_dim

        x = self.embed[input_ids].half()                       # [B, T, C]
        if cfg.norm_first:
            x = F.layer_norm(x, (C,), self.pre_norm_w, self.pre_norm_b, cfg.norm_eps)

        v_first: torch.Tensor | None = None
        if state_out is not None:
            new_attn_prev = state_out.attn_prev
            new_ffn_prev = state_out.ffn_prev
            new_wkv = state_out.wkv
        else:
            new_attn_prev = torch.empty_like(state.attn_prev)
            new_ffn_prev = torch.empty_like(state.ffn_prev)
            new_wkv = torch.empty_like(state.wkv)

        for i, lp in enumerate(self.layers):
            residual = x
            h = F.layer_norm(residual, (C,), lp.attn_norm_w, lp.attn_norm_b, cfg.norm_eps)
            # In-batch token shift; the first token continues from the saved
            # state. Read the old state BEFORE overwriting it (in-place safe).
            h_prev = torch.cat([state.attn_prev[i].view(1, 1, C).expand(B, 1, C),
                                h[:, :-1, :]], dim=1)
            new_attn_prev[i] = h.reshape(B * T, C)[B * T - 1]
            delta = h_prev - h                                  # [B, T, C]
            xs = [torch.empty_like(h) for _ in range(6)]
            torch._foreach_copy_(xs, [h] * 6)
            # NB: the trailing underscore matters -- the functional
            # _foreach_addcmul returns new tensors and silently drops the shift.
            torch._foreach_addcmul_(xs, [delta] * 6, list(lp.x))
            xr, xw, xk, xv, xa, xg = xs

            r = self._mm(xr, lp.r_w)
            k = self._mm(xk, lp.k_w)
            v = self._mm(xv, lp.v_w)

            if B == 1 and T == 1 and _FUSED:
                # ---------------- fused decode fast path ----------------
                # LoRA W1 GEMVs (skinny) via cuBLAS; the four W2 GEMVs +
                # activations run in ONE fused kernel.
                # W1 GEMVs (skinny) via cuBLAS, with the h1-side activation
                # folded in (tanh for w, sigmoid for g); the W2 GEMV +
                # output activation + bias + scale all run in the split-R
                # gate kernel.
                if i == 0:
                    v_first = v
                    xv_use = None
                else:
                    xv_use = xv.view(-1)
                h1w, h1a, h1g, h1v = lora_h1(
                    (xw.view(-1), xa.view(-1), xg.view(-1), xv_use),
                    (lp.w_l0, lp.a_l0, lp.g_l0, lp.v_l0),
                    acts=(1, 0, 2, 0), phases=_LORA_PHASES)
                w, a, g, t = lora_gates(
                    (h1w, h1a, h1g, h1v),
                    (lp.w_l1, lp.a_l1, lp.g_l1, lp.v_l1),
                    (lp.w_lb, lp.a_lb, lp.v_lb),
                    cfg.w_scale, _LORA_SPLITR)
                if i != 0:
                    v = v + (v_first - v) * t.view(1, 1, C)
                kk = F.normalize((k * lp.k_k).half().view(1, 1, H, D), p=2.0,
                                 dim=-1, eps=1e-12).view(1, 1, C)
                k2 = (k * (1.0 + (a - 1.0) * lp.k_a)).half()
                o = wkv7(r.reshape(1, C), w.view(1, C), k2.reshape(1, C),
                         v.reshape(1, C), (-kk).reshape(1, C),
                         (kk * a).reshape(1, C),
                         state.wkv[i], new_wkv[i]).view(1, 1, C)      # fp32
                og = wkv7_post(o.view(-1), k2.view(-1), r.view(-1),
                               v.view(-1), g.view(-1), lp.gn_w, lp.gn_b,
                               lp.r_k, H, D, cfg.group_norm_eps).view(1, 1, C)
                x = residual + self._mm(og, lp.o_w)
            else:
                # ---------------- prefill / unfused path ----------------
                h1 = self._mm(xw, lp.w_l0)
                w = (cfg.w_scale * torch.sigmoid(
                    torch.tanh(h1) @ lp.w_l1 + lp.w_lb
                )).half()
                h2 = self._mm(xa, lp.a_l0)
                a = torch.sigmoid(
                    h2 @ lp.a_l1 + lp.a_lb
                ).half()
                if i == 0:
                    v_first = v
                else:
                    h3 = self._mm(xv, lp.v_l0)
                    t = torch.sigmoid(
                        h3 @ lp.v_l1 + lp.v_lb
                    )
                    v = v + (v_first - v) * t
                kk = F.normalize((k * lp.k_k).half().view(B, T, H, D), p=2.0,
                                 dim=-1, eps=1e-12).view(B, T, C)
                k = (k * (1.0 + (a - 1.0) * lp.k_a)).half()

                o = wkv7(r.reshape(B * T, C).contiguous(),
                         w.reshape(B * T, C).contiguous(),
                         k.reshape(B * T, C).contiguous(),
                         v.reshape(B * T, C).contiguous(),
                         (-kk).reshape(B * T, C).contiguous(),
                         (kk * a).reshape(B * T, C).contiguous(),
                         state.wkv[i], new_wkv[i]).reshape(B, T, C)     # fp32

                o = F.group_norm(o.reshape(B * T, C), H,
                                 lp.gn_w.float(), lp.gn_b.float(),
                                 cfg.group_norm_eps).reshape(B, T, C)
                rk = (k.view(B, T, H, D) * r.view(B, T, H, D)
                      * lp.r_k.view(H, D)).sum(-1)                  # [B, T, H]
                o = o + (rk.view(B, T, H, 1) * v.view(B, T, H, D)).reshape(B, T, C)
                g = torch.sigmoid(self._mm(xg, lp.g_l0)) @ lp.g_l1
                x = residual + self._mm((o * g).half(), lp.o_w)

            residual = x
            h = F.layer_norm(residual, (C,), lp.ffn_norm_w, lp.ffn_norm_b, cfg.norm_eps)
            # Read the old state BEFORE overwriting it (in-place safe).
            h_prev = torch.cat([state.ffn_prev[i].view(1, 1, C).expand(B, 1, C),
                                h[:, :-1, :]], dim=1)
            new_ffn_prev[i] = h.reshape(B * T, C)[B * T - 1]
            delta = h_prev - h                                  # [B, T, C]
            hf = torch.addcmul(h, delta, lp.ffn_x_k)
            f = self._mm(hf, lp.ffn_key)
            x = residual + self._mm(sqrelu(f), lp.ffn_value)
            if capture is not None:
                capture.append(x.float().cpu())

        x = F.layer_norm(x, (C,), self.final_norm_w, self.final_norm_b, cfg.norm_eps)
        logits = (x @ self.lm_head).float() if x.shape[1] > 1 else \
            gemv16(self.lm_head, x.view(-1), out_f32=True).view(1, 1, -1)
        if state_out is not None:
            return logits, state_out
        return logits, RWKV7State(new_wkv, new_attn_prev, new_ffn_prev)

    # ------------------------------------------------------ CUDA-graph decode
    def _ensure_decode_graph(self):
        """Capture a CUDA graph for 1-token decode (state updated in-place).

        Returns (graph, logits_buf [1,1,V] fp32, state, ids_buf [1,1] int64).
        The graph is captured once and reused; each generation copies the
        prefill state into ``state`` before replaying.
        """
        if getattr(self, "_dec_graph", None) is None:
            state = self.init_state()
            self._dec_ids = torch.zeros(1, 1, dtype=torch.int64, device=self.device)
            for _ in range(2):  # warm up (cuBLAS workspaces, allocator)
                self.forward(self._dec_ids, state, state_out=state)
            torch.cuda.synchronize(self.device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                logits, _ = self.forward(self._dec_ids, state, state_out=state)
            self._dec_graph = g
            self._dec_state = state
            self._dec_logits = logits
        return self._dec_graph, self._dec_logits, self._dec_state, self._dec_ids

    # ---------------------------------------------------------------- greedy
    def greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        stop_ids: set[int] | None = None,
        stop_at_first: bool = True,
        callback=None,
        use_graph: bool = True,
    ) -> list[int]:
        """Greedy decode from input_ids [1, T]. Returns generated token ids."""
        stop_ids = stop_ids if stop_ids is not None else {0}
        state = self.init_state()
        logits, state = self.forward(input_ids, state)
        last = logits[0, -1].argmax().item()
        out = [last]
        if callback:
            callback(0, last)
        if (last in stop_ids and stop_at_first) or max_tokens <= 1:
            return out
        if use_graph:
            g, logit_buf, gstate, ids_buf = self._ensure_decode_graph()
            for a, b in zip(state.wkv, gstate.wkv):
                b.copy_(a)
            for a, b in zip(state.attn_prev, gstate.attn_prev):
                b.copy_(a)
            for a, b in zip(state.ffn_prev, gstate.ffn_prev):
                b.copy_(a)
            ids_buf.fill_(last)
            for step in range(1, max_tokens):
                g.replay()
                last = logit_buf[0, 0].argmax().item()
                out.append(last)
                if callback:
                    callback(step, last)
                if last in stop_ids and stop_at_first:
                    break
                ids_buf.fill_(last)
            return out
        for step in range(max_tokens - 1):
            cur = input_ids.new_full((1, 1), last)
            logits, state = self.forward(cur, state)
            last = logits[0, 0].argmax().item()
            out.append(last)
            if callback:
                callback(step + 1, last)
            if last in stop_ids and stop_at_first:
                break
        return out
