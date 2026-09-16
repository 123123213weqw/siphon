"""Compare pure-torch fp32 reference (torch_ref) logits vs a saved ref logits file,
per token position. Decides whether the divergence is in the formula (torch_ref)
or only in the CUDA kernel (my model).

Usage:
  python -m rwkv7_engine.ref_vs_ref --model-dir D --ref-pt /tmp/ref_15b.pt --prompt "..."
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from . import torch_ref as TR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref-pt", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    import json
    cfgd = json.load(open(os.path.join(args.model_dir, "config.json")))
    cfg = {
        "vocab_size": cfgd["vocab_size"], "hidden_size": cfgd["hidden_size"],
        "num_heads": cfgd["num_heads"], "head_dim": cfgd.get("head_dim", 64),
        "num_layers": cfgd.get("num_hidden_layers", cfgd.get("num_layers", 32)),
        "intermediate_size": cfgd["intermediate_size"],
    }
    ids = TR._prompt_ids(args.model_dir, args.prompt)
    W = TR._load_weights(args.model_dir)
    logits = TR.forward(W, cfg, ids)   # [1, T, V] fp32

    ref = torch.load(args.ref_pt, map_location="cpu")
    A = ref["all_logits"]              # [T, V] (HF native, fp16->fp32)
    T = min(logits.shape[1], A.shape[0])
    print(f"prompt ids: {ids}  (torch_ref T={logits.shape[1]}, ref T={A.shape[0]})")
    for t in range(T):
        o = logits[0, t].float()
        d = (A[t] - o).abs()
        cos = F.cosine_similarity(A[t].unsqueeze(0), o.unsqueeze(0)).item()
        print(f"tok{t} max_abs={d.max().item():8.3f} mean_abs={d.mean().item():7.4f} "
              f"cos={cos:.6f} ref={A[t].argmax().item()} torchref={o.argmax().item()}")


if __name__ == "__main__":
    main()
