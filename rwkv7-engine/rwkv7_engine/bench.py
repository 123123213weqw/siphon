"""End-to-end benchmark for the Siphon RWKV7 engine.

Measures, on one GPU:
  1. cold load   - wall time to load all weights to the GPU (fresh process)
  2. prefill     - tokens/s for a fixed prompt (chunked forward pass)
  3. decode      - tokens/s for greedy generation

Usage:
  python -m rwkv7_engine.bench --model-dir /path/to/g1j-hf --device cuda:0 \
      --prefill-tokens 2048 --decode-tokens 128 --warmup 3 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from .model import RWKV7Model
from .tokenizer import RWKVTrieTokenizer


def _bench_load(model_dir: str, device: str, loader: str) -> dict:
    t0 = time.perf_counter()
    model = RWKV7Model.from_hf_dir(model_dir, device=device, loader=loader)
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    nbytes = sum(t.numel() * t.element_size()
                 for t in [model.embed, model.lm_head]
                 for _ in [0])
    # count all params
    total = 0
    for lp in model.layers:
        for v in lp.__dict__.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    for n in ("embed", "lm_head", "pre_norm_w", "pre_norm_b", "final_norm_w", "final_norm_b"):
        t = getattr(model, n)
        if torch.is_tensor(t):
            total += t.numel() * t.element_size()
    return {"seconds": dt, "gb": total / 1e9, "gb_per_s": total / 1e9 / dt, "loader": loader}


def _prompt_ids(model_dir: str, n_tokens: int) -> list[int]:
    """Deterministic prompt: repeat a fixed text until n_tokens is reached."""
    vocab_file = os.path.join(model_dir, "rwkv_vocab_v20230424.txt")
    tok = RWKVTrieTokenizer(vocab_file)
    text = ("The quick brown fox jumps over the lazy dog. "
            "In a quiet room, a single lamp glows softly on the wooden desk. "
            "Outside, the rain taps gently against the window glass, a steady "
            "rhythm that fills the silence with sound. ")
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(tok.encode(text))
    return ids[:n_tokens]


def _bench_prefill(model: RWKV7Model, ids: list[int], device: str,
                   warmup: int, chunk: int = 512) -> dict:
    x = torch.tensor([ids], dtype=torch.int64, device=device)
    T = x.shape[1]
    state = model.init_state()
    for _ in range(warmup):
        model.forward(x, model.init_state())
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    logits, state = model.forward(x, model.init_state())
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    del logits, state
    torch.cuda.empty_cache()
    return {"tokens": T, "seconds": dt, "tok_per_s": T / dt}


def _bench_decode(model: RWKV7Model, ids: list[int], device: str,
                  n_tokens: int, warmup: int) -> dict:
    x = torch.tensor([ids], dtype=torch.int64, device=device)
    for _ in range(warmup):
        model.greedy_generate(x, min(8, n_tokens))
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = model.greedy_generate(x, n_tokens)
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    n = len(out) - 1  # exclude the first token from the prefill
    return {"tokens": n, "seconds": dt, "tok_per_s": n / dt if dt > 0 else 0.0,
            "generated": out[:32]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--loader", default="siphon", choices=["siphon", "safetensors"])
    ap.add_argument("--prefill-tokens", type=int, default=2048)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    result: dict = {
        "device": torch.cuda.get_device_name(args.device),
        "model_dir": args.model_dir,
        "loader": args.loader,
    }

    if not args.skip_load:
        result["load"] = _bench_load(args.model_dir, args.device, args.loader)

    model = RWKV7Model.from_hf_dir(args.model_dir, device=args.device, loader=args.loader)
    ids = _prompt_ids(args.model_dir, max(args.prefill_tokens, 64))

    if not args.skip_prefill:
        result["prefill"] = _bench_prefill(model, ids, args.device, args.warmup)
    if not args.skip_decode:
        result["decode"] = _bench_decode(model, ids[:512], args.device,
                                         args.decode_tokens, args.warmup)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
