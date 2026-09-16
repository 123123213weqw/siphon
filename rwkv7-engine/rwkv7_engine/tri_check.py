"""Three-way per-position comparison: HF-native ref, Siphon model, pure-torch fp32 ref.

Usage:
  python -m rwkv7_engine.tri_check --model-dir D --ref-pt A --ours-pt B --prompt "..."
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from . import torch_ref as TR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref-pt", required=True)
    ap.add_argument("--ours-pt", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    cfgd = json.load(open(os.path.join(args.model_dir, "config.json")))
    cfg = {
        "vocab_size": cfgd["vocab_size"], "hidden_size": cfgd["hidden_size"],
        "num_heads": cfgd["num_heads"], "head_dim": cfgd.get("head_dim", 64),
        "num_layers": cfgd.get("num_hidden_layers", cfgd.get("num_layers", 32)),
        "intermediate_size": cfgd["intermediate_size"],
    }
    ids = TR._prompt_ids(args.model_dir, args.prompt)
    W = TR._load_weights(args.model_dir)
    tr = TR.forward(W, cfg, ids)[0].float()      # [T, V] torch_ref fp32

    ref = torch.load(args.ref_pt, map_location="cpu")["all_logits"].float()
    ours = torch.load(args.ours_pt, map_location="cpu")["all_logits"].float()
    T = min(ref.shape[0], ours.shape[0], tr.shape[0])
    for t in range(T):
        row = []
        for name, m in (("ref", ref[t]), ("model", ours[t]), ("tref", tr[t])):
            row.append(f"{name}:{m.argmax().item()}")
        cm = F.cosine_similarity(ours[t].unsqueeze(0), tr[t].unsqueeze(0)).item()
        cr = F.cosine_similarity(ref[t].unsqueeze(0), tr[t].unsqueeze(0)).item()
        print(f"tok{t} " + "  ".join(row) + f"  cos(model,tref)={cm:.5f} cos(ref,tref)={cr:.5f}")


if __name__ == "__main__":
    main()
