"""Greedy token-by-token agreement between the HF reference and Siphon model.

Generates N tokens from a prompt with both implementations (greedy, temp=0)
and reports the first divergence and the agreement rate.

Usage:
  python -m rwkv7_engine.greedy_check --model-dir D --prompt "..." --n 32
"""
from __future__ import annotations

import argparse
import os

import torch


def _prompt_ids(model_dir: str, prompt: str) -> list[int]:
    from .tokenizer import RWKVTrieTokenizer
    tok = RWKVTrieTokenizer(os.path.join(model_dir, "rwkv_vocab_v20230424.txt"))
    return [1] + tok.encode(prompt)


def gen_ref(model_dir: str, device: str, ids: list[int], n: int):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    model.eval()
    x = torch.tensor([ids], device=device)
    toks = list(ids)
    with torch.no_grad():
        for _ in range(n):
            out = model(x, use_cache=False)
            nxt = out.logits[0, -1].argmax().item()
            toks.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return toks


def gen_ours(model_dir: str, device: str, ids: list[int], n: int):
    from .model import RWKV7Model
    model = RWKV7Model.from_hf_dir(model_dir, device=device, loader="safetensors")
    state = model.init_state()
    x = torch.tensor([ids], device=device)
    logits, state = model.forward(x, state)
    toks = list(ids)
    for _ in range(n):
        nxt = logits[0, -1].argmax().item()
        toks.append(nxt)
        step = torch.tensor([[nxt]], device=device)
        logits, state = model.forward(step, state)
    return toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    ids = _prompt_ids(args.model_dir, args.prompt)

    ref = gen_ref(args.model_dir, args.device, ids, args.n)
    torch.cuda.empty_cache()
    ours = gen_ours(args.model_dir, args.device, ids, args.n)

    m = min(len(ref), len(ours))
    agree = sum(1 for i in range(m) if ref[i] == ours[i])
    first_diff = next((i for i in range(m) if ref[i] != ours[i]), None)
    print(f"prompt: {args.prompt!r}")
    print(f"agreement: {agree}/{m} = {agree / m:.3f}")
    print(f"first divergence at index {first_diff}")
    print("ref :", ref)
    print("ours:", ours)


if __name__ == "__main__":
    main()
