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

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RWKV7Config
from .wkv import wkv7


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
    w_l1: torch.Tensor                 # [C, rank]
    w_lb: torch.Tensor                 # [C]
    a_l0: torch.Tensor
    a_l1: torch.Tensor
    a_lb: torch.Tensor
    v_l0: torch.Tensor | None          # None at layer 0
    v_l1: torch.Tensor | None
    v_lb: torch.Tensor | None
    g_l0: torch.Tensor                 # [rank_g, C]
    g_l1: torch.Tensor                 # [C, rank_g]
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
    lm_head: torch.Tensor = field(repr=False, default=None)  # [V, C] fp16
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
        model.lm_head = get("lm_head.weight")

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
                v_l0 = get(p + "attn.v_lora.lora.0.weight")
                v_l1 = get(p + "attn.v_lora.lora.2.weight").t()
                v_lb = getvec(p + "attn.v_lora.lora.2.bias")
            model.layers.append(_LayerParams(
                attn_norm_w=getvec(p + "attn_norm.weight"),
                attn_norm_b=getvec(p + "attn_norm.bias"),
                x=x,
                r_w=get(p + "attn.r_proj.weight").t(),
                k_w=get(p + "attn.k_proj.weight").t(),
                v_w=get(p + "attn.v_proj.weight").t(),
                o_w=get(p + "attn.o_proj.weight").t(),
                w_l0=get(p + "attn.w_lora.lora.0.weight"),
                w_l1=get(p + "attn.w_lora.lora.2.weight").t(),
                w_lb=getvec(p + "attn.w_lora.lora.2.bias"),
                a_l0=get(p + "attn.a_lora.lora.0.weight"),
                a_l1=get(p + "attn.a_lora.lora.2.weight").t(),
                a_lb=getvec(p + "attn.a_lora.lora.2.bias"),
                v_l0=v_l0, v_l1=v_l1, v_lb=v_lb,
                g_l0=get(p + "attn.g_lora.lora.0.weight"),
                g_l1=get(p + "attn.g_lora.lora.2.weight").t(),
                k_k=getvec(p + "attn.k_k"),
                k_a=getvec(p + "attn.k_a"),
                r_k=getvec(p + "attn.r_k"),
                gn_w=getvec(p + "attn.g_norm.weight"),
                gn_b=getvec(p + "attn.g_norm.bias"),
                ffn_norm_w=getvec(p + "ffn_norm.weight"),
                ffn_norm_b=getvec(p + "ffn_norm.bias"),
                ffn_x_k=getvec(p + "ffn.x_k"),
                ffn_key=get(p + "ffn.key.weight").t(),
                ffn_value=get(p + "ffn.value.weight").t(),
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
    def forward(self, input_ids: torch.Tensor, state: RWKV7State,
                capture: list | None = None) -> tuple[torch.Tensor, RWKV7State]:
        """input_ids: [B, T] int64 -> (logits [B, T, V] fp32, new_state)."""
        B, T = input_ids.shape
        cfg = self.config
        C = cfg.hidden_size
        H, D = cfg.num_heads, cfg.head_dim

        x = self.embed[input_ids].half()                       # [B, T, C]
        if cfg.norm_first:
            x = F.layer_norm(x, (C,), self.pre_norm_w, self.pre_norm_b, cfg.norm_eps)

        v_first: torch.Tensor | None = None
        new_attn_prev = torch.empty_like(state.attn_prev)
        new_ffn_prev = torch.empty_like(state.ffn_prev)
        new_wkv = torch.empty_like(state.wkv)

        for i, lp in enumerate(self.layers):
            residual = x
            h = F.layer_norm(residual, (C,), lp.attn_norm_w, lp.attn_norm_b, cfg.norm_eps)
            new_attn_prev[i] = h.reshape(B * T, C)[B * T - 1]

            # In-batch token shift; the first token continues from the saved state.
            h_prev = torch.cat([state.attn_prev[i].view(1, 1, C).expand(B, 1, C),
                                h[:, :-1, :]], dim=1)
            delta = h_prev - h                                  # [B, T, C]
            xr = h + delta * lp.x[0]
            xw = h + delta * lp.x[1]
            xk = h + delta * lp.x[2]
            xv = h + delta * lp.x[3]
            xa = h + delta * lp.x[4]
            xg = h + delta * lp.x[5]

            r = torch.matmul(xr, lp.r_w)
            k = torch.matmul(xk, lp.k_w)
            v = torch.matmul(xv, lp.v_w)

            w = (cfg.w_scale * torch.sigmoid(
                torch.matmul(torch.tanh(torch.matmul(xw, lp.w_l0.t())), lp.w_l1) + lp.w_lb
            )).half()
            a = torch.sigmoid(
                torch.matmul(xa, lp.a_l0.t()) @ lp.a_l1 + lp.a_lb
            ).half()
            if i == 0:
                v_first = v
            else:
                t = torch.sigmoid(
                    torch.matmul(xv, lp.v_l0.t()) @ lp.v_l1 + lp.v_lb
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
            g = torch.matmul(torch.sigmoid(torch.matmul(xg, lp.g_l0.t())), lp.g_l1)
            x = residual + torch.matmul((o * g).half(), lp.o_w)

            residual = x
            h = F.layer_norm(residual, (C,), lp.ffn_norm_w, lp.ffn_norm_b, cfg.norm_eps)
            new_ffn_prev[i] = h.reshape(B * T, C)[B * T - 1]
            h_prev = torch.cat([state.ffn_prev[i].view(1, 1, C).expand(B, 1, C),
                                h[:, :-1, :]], dim=1)
            delta = h_prev - h                                  # [B, T, C]
            hf = h + delta * lp.ffn_x_k
            x = residual + torch.matmul(sqrelu(torch.matmul(hf, lp.ffn_key)), lp.ffn_value)
            if capture is not None:
                capture.append(x.float().cpu())

        x = F.layer_norm(x, (C,), self.final_norm_w, self.final_norm_b, cfg.norm_eps)
        logits = (x @ self.lm_head.t()).float()
        return logits, RWKV7State(new_wkv, new_attn_prev, new_ffn_prev)

    # ---------------------------------------------------------------- greedy
    def greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        stop_ids: set[int] | None = None,
        stop_at_first: bool = True,
        callback=None,
    ) -> list[int]:
        """Greedy decode from input_ids [1, T]. Returns generated token ids."""
        stop_ids = stop_ids if stop_ids is not None else {0}
        state = self.init_state()
        logits, state = self.forward(input_ids, state)
        last = logits[0, -1].argmax().item()
        out = [last]
        if callback:
            callback(0, last)
        if last in stop_ids and stop_at_first:
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
