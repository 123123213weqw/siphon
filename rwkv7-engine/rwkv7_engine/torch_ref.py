"""Pure-torch FP32 reference for the RWKV7 (G1x) forward.

A clean, dependency-light implementation used to (a) validate the Siphon
engine's numerics and (b) check whether the HF native fused model follows the
canonical formula. Everything runs in float32; the Wkv recurrence is a plain
Python loop (no custom kernel).

Usage:
  python -m rwkv7_engine.torch_ref --model-dir D --prompt "..." [--layers]
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import torch
import torch.nn.functional as F

W_SCALE = -0.6065306597126334
GN_EPS = 64e-5
L2_EPS = 1e-12
NORM_EPS = 1e-5


def _prompt_ids(model_dir: str, prompt: str) -> list[int]:
    from .tokenizer import RWKVTrieTokenizer
    tok = RWKVTrieTokenizer(os.path.join(model_dir, "rwkv_vocab_v20230424.txt"))
    return [1] + tok.encode(prompt)


def _load_weights(model_dir: str) -> dict:
    import glob
    fs = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    W: dict[str, torch.Tensor] = {}
    for f in fs:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            off = 8 + n
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                lo, hi = meta["data_offsets"]
                fh.seek(off + lo)
                raw = fh.read(hi - lo)
                dt = meta["dtype"]
                if dt == "F16":
                    t = torch.frombuffer(bytearray(raw), dtype=torch.float16)
                elif dt == "F32":
                    t = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                elif dt == "I64":
                    t = torch.frombuffer(bytearray(raw), dtype=torch.int64)
                else:
                    raise ValueError(dt)
                shape = meta["shape"]
                if len(shape) == 0:
                    t = t.reshape(())
                elif len(shape) == 1:
                    t = t.reshape(shape[0])
                else:
                    t = t.reshape(*shape)
                W[k] = t.float().contiguous()
    return W


def _wkv_loop(r, w, k, v, a, b, state):
    """r,w,k,v,a,b: [T, C] fp32; state: [H, D, D] fp32 (row i per head). Returns [T, C] and new state."""
    T, C = r.shape
    D = state.shape[-1]
    H = C // D
    out = torch.empty(T, C, dtype=torch.float32)
    for t in range(T):
        rt = r[t].view(H, D)
        wt = torch.exp(w[t].view(H, D))
        kt = k[t].view(H, D)
        vt = v[t].view(H, D)
        at = a[t].view(H, D)
        bt = b[t].view(H, D)
        sa = (at.unsqueeze(1) * state).sum(-1)                     # [H, D]  a . row_i
        state = (state * wt.unsqueeze(1)
                 + kt.unsqueeze(1) * vt.unsqueeze(2)
                 + sa.unsqueeze(2) * bt.unsqueeze(1))
        out[t] = (state * rt.unsqueeze(1)).sum(-1).reshape(C)
    return out, state


def forward(W: dict, cfg: dict, ids: list[int], capture: list | None = None):
    dev = "cpu"
    V, C = cfg["vocab_size"], cfg["hidden_size"]
    H, D = cfg["num_heads"], cfg["head_dim"]
    L, I = cfg["num_layers"], cfg["intermediate_size"]

    def get(name):
        return W[name].to(dev)

    embed = get("model.embeddings.weight")
    lm_head = get("lm_head.weight")
    pre_w = get("model.layers.0.pre_norm.weight") if f"model.layers.0.pre_norm.weight" in W else None
    pre_b = get("model.layers.0.pre_norm.bias") if f"model.layers.0.pre_norm.bias" in W else None

    x = embed[torch.tensor(ids)].unsqueeze(0)             # [1, T, C] fp32
    if pre_w is not None:
        x = F.layer_norm(x, (C,), pre_w, pre_b, NORM_EPS)

    T = x.shape[1]
    attn_prev = [torch.zeros(C) for _ in range(L)]
    ffn_prev = [torch.zeros(C) for _ in range(L)]
    wkv_state = [torch.zeros(H, D, D) for _ in range(L)]

    v_first = None
    for i in range(L):
        p = f"model.layers.{i}."
        residual = x
        h = F.layer_norm(residual, (C,), get(p + "attn_norm.weight"),
                         get(p + "attn_norm.bias"), NORM_EPS)
        attn_prev[i] = h[0, -1].clone()
        h_prev = torch.cat([attn_prev[i].view(1, 1, C), h[:, :-1, :]], dim=1)
        delta = h_prev - h
        xr = h + delta * get(p + "attn.x_r")
        xw = h + delta * get(p + "attn.x_w")
        xk = h + delta * get(p + "attn.x_k")
        xv = h + delta * get(p + "attn.x_v")
        xa = h + delta * get(p + "attn.x_a")
        xg = h + delta * get(p + "attn.x_g")

        r = xr @ get(p + "attn.r_proj.weight").t()
        k = xk @ get(p + "attn.k_proj.weight").t()
        v = xv @ get(p + "attn.v_proj.weight").t()

        w = W_SCALE * torch.sigmoid(torch.tanh(xw @ get(p + "attn.w_lora.lora.0.weight").t())
                                    @ get(p + "attn.w_lora.lora.2.weight").t()
                                    + get(p + "attn.w_lora.lora.2.bias"))
        a = torch.sigmoid(xa @ get(p + "attn.a_lora.lora.0.weight").t()
                          @ get(p + "attn.a_lora.lora.2.weight").t()
                          + get(p + "attn.a_lora.lora.2.bias"))
        if i == 0:
            v_first = v
        else:
            t = torch.sigmoid(xv @ get(p + "attn.v_lora.lora.0.weight").t()
                              @ get(p + "attn.v_lora.lora.2.weight").t()
                              + get(p + "attn.v_lora.lora.2.bias"))
            v = v + (v_first - v) * t
        kk = F.normalize((k * get(p + "attn.k_k")).view(1, T, H, D), p=2.0,
                         dim=-1, eps=L2_EPS).view(1, T, C)
        k = k * (1.0 + (a - 1.0) * get(p + "attn.k_a"))

        o, wkv_state[i] = _wkv_loop(r[0], w[0], k[0], v[0],
                                    (-kk)[0], (kk * a)[0], wkv_state[i])
        o = o.unsqueeze(0)                                  # [1, T, C]
        o = F.group_norm(o.reshape(T, C), H, get(p + "attn.g_norm.weight"),
                         get(p + "attn.g_norm.bias"), GN_EPS).reshape(1, T, C)
        rk = (k.view(1, T, H, D) * r.view(1, T, H, D)
              * get(p + "attn.r_k").view(H, D)).sum(-1)     # [1, T, H]
        o = o + (rk.view(1, T, H, 1) * v.view(1, T, H, D)).reshape(1, T, C)
        g = torch.sigmoid(xg @ get(p + "attn.g_lora.lora.0.weight").t()) \
            @ get(p + "attn.g_lora.lora.2.weight").t()
        x = residual + (o * g) @ get(p + "attn.o_proj.weight").t()

        h = F.layer_norm(x, (C,), get(p + "ffn_norm.weight"),
                         get(p + "ffn_norm.bias"), NORM_EPS)
        ffn_prev[i] = h[0, -1].clone()
        h_prev = torch.cat([ffn_prev[i].view(1, 1, C), h[:, :-1, :]], dim=1)
        delta = h_prev - h
        hf = h + delta * get(p + "ffn.x_k")
        key = hf @ get(p + "ffn.key.weight").t()
        kact = key * torch.relu(key)
        x = x + kact @ get(p + "ffn.value.weight").t()
        if capture is not None:
            capture.append(x.clone())

    x = F.layer_norm(x, (C,), get("model.norm.weight"), get("model.norm.bias"), NORM_EPS)
    logits = x @ lm_head.t()
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()
    import json as _json
    cfgd = _json.load(open(os.path.join(args.model_dir, "config.json")))
    cfg = {
        "vocab_size": cfgd["vocab_size"], "hidden_size": cfgd["hidden_size"],
        "num_heads": cfgd["num_heads"], "head_dim": cfgd.get("head_dim", 64),
        "num_layers": cfgd.get("num_hidden_layers", cfgd.get("num_layers", 32)),
        "intermediate_size": cfgd["intermediate_size"],
    }
    ids = _prompt_ids(args.model_dir, args.prompt)
    W = _load_weights(args.model_dir)
    logits = forward(W, cfg, ids)
    print(json.dumps({
        "prompt": args.prompt, "ids": ids,
        "last_top5": logits[0, -1].topk(5).indices.tolist(),
        "last_top5_vals": logits[0, -1].topk(5).values.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
