"""Phase runner for the split correctness check.

Phase ref : load the HF/FLA remote-code model, run a prompt, save last-token
            logits + top-k to disk.
Phase ours: load via the Siphon RWKV7Model, run the same prompt, save the same.
Phase cmp : load both files from disk and report the stats.

Usage:
  python -m rwkv7_engine.split_check --phase ref  --model-dir D --prompt "..." --out ref.pt
  python -m rwkv7_engine.split_check --phase ours --model-dir D --prompt "..." --out ours.pt
  python -m rwkv7_engine.split_check --phase cmp  --a ref.pt --b ours.pt
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def _prompt_ids(model_dir: str, prompt: str) -> list[int]:
    from .tokenizer import RWKVTrieTokenizer
    tok = RWKVTrieTokenizer(os.path.join(model_dir, "rwkv_vocab_v20230424.txt"))
    return [1] + tok.encode(prompt)


def run_ref(model_dir: str, device: str, prompt: str, out: str) -> None:
    from transformers import AutoModelForCausalLM
    ids = _prompt_ids(model_dir, prompt)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    model.eval()
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(x, use_cache=False).logits
    payload = {
        "prompt": prompt,
        "ids": ids,
        "last_logits": logits[0, -1].float().cpu(),
        "all_logits": logits[0].float().cpu(),
    }
    torch.save(payload, out)
    print(f"saved ref logits to {out} (last-token top5: {logits[0, -1].topk(5).indices.tolist()})")


def run_ours(model_dir: str, device: str, prompt: str, out: str, loader: str = "siphon") -> None:
    from .model import RWKV7Model
    ids = _prompt_ids(model_dir, prompt)
    model = RWKV7Model.from_hf_dir(model_dir, device=device, loader=loader)
    x = torch.tensor([ids], device=device)
    logits, _ = model.forward(x, model.init_state())
    payload = {
        "prompt": prompt,
        "ids": ids,
        "last_logits": logits[0, -1].float().cpu(),
        "all_logits": logits[0].float().cpu(),
    }
    torch.save(payload, out)
    print(f"saved our logits to {out} (last-token top5: {logits[0, -1].topk(5).indices.tolist()})")


def cmp(a: str, b: str) -> None:
    pa = torch.load(a, map_location="cpu")
    pb = torch.load(b, map_location="cpu")
    la, lb = pa["last_logits"], pb["last_logits"]
    diff = (la - lb).abs()
    rel = diff / lb.abs().clamp_min(1.0)
    cos = torch.nn.functional.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item()
    ta, tb = la.topk(10).indices.tolist(), lb.topk(10).indices.tolist()
    # full-sequence stats
    A, B = pa["all_logits"], pb["all_logits"]
    n = min(A.shape[1], B.shape[1])
    Adiff = (A[:, :n] - B[:, :n]).abs()
    print(json.dumps({
        "last_max_abs_diff": diff.max().item(),
        "last_mean_abs_diff": diff.mean().item(),
        "last_max_rel_diff": rel.max().item(),
        "last_cosine": cos,
        "top1_match": ta[0] == tb[0],
        "top10_overlap": len(set(ta) & set(tb)) / 10.0,
        "seq_max_abs_diff": Adiff.max().item(),
        "seq_mean_abs_diff": Adiff.mean().item(),
        "seq_cosine_mean": torch.nn.functional.cosine_similarity(
            A[:, :n].flatten(1), B[:, :n].flatten(1)).mean().item(),
        "top10_ref": ta[:5],
        "top10_ours": tb[:5],
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["ref", "ours", "cmp"])
    ap.add_argument("--model-dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--loader", default="siphon", choices=["siphon", "safetensors"])
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    if args.phase == "ref":
        run_ref(args.model_dir, args.device, args.prompt, args.out)
    elif args.phase == "ours":
        run_ours(args.model_dir, args.device, args.prompt, args.out, args.loader)
    else:
        cmp(args.a, args.b)


if __name__ == "__main__":
    main()
