"""Correctness comparison for the Siphon RWKV7 engine.

Two independent references:
  * HF/FLA   - the transformers remote-code model (FLA kernels) loaded from the
               same HF dir; compares per-token logits.
  * llama.cpp - llama-server (or llama-cli) on the GGUF build; compares greedy
               generated token ids.

Usage:
  python -m rwkv7_engine.compare --model-dir DIR --ref hf --prompt "Hello" \
      --prompt-file prompts.txt --max-tokens 128
  python -m rwkv7_engine.compare --model-dir DIR --ref llamacpp \
      --server-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request

import torch

from .model import RWKV7Model
from .tokenizer import RWKVTrieTokenizer

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village, there lived",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The quick brown fox jumps over the lazy dog.",
]


def _vocab_path(model_dir: str) -> str:
    for name in ("rwkv_vocab_v20230424.txt",):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("rwkv_vocab_v20230424.txt not in " + model_dir)


def _load_hf_reference(model_dir: str, device: str):
    """Load the HF/FLA remote-code model. Returns a callable (ids [1,T]) -> logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    model.eval()

    def run(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(x, use_cache=False)
        return out.logits

    return run, tok


def _logit_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1.0)
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": (diff / denom).max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(1), b.flatten(1), dim=1).mean().item(),
    }


def compare_hf(model: RWKV7Model, model_dir: str, device: str,
               prompts: list[str], max_tokens: int, n_layers: int | None) -> dict:
    hf_run, _ = _load_hf_reference(model_dir, device)
    tok = RWKVTrieTokenizer(_vocab_path(model_dir))
    rows = []
    for prompt in prompts:
        ids = [1] + tok.encode(prompt)  # BOS = 1
        x = torch.tensor([ids], device=device)
        ours, _ = model.forward(x, model.init_state())
        ref = hf_run(ids)
        T = min(ours.shape[1], ref.shape[1])
        # compare the last n_layers layers' worth of positions is not exposed;
        # compare all T logits
        stats = _logit_stats(ours[:, :T], ref[:, :T])
        ours_top = ours[0, -1].topk(5).indices.tolist()
        ref_top = ref[0, -1].topk(5).indices.tolist()
        rows.append({
            "prompt": prompt[:60],
            "tokens": T,
            **stats,
            "top5_ours": ours_top,
            "top5_ref": ref_top,
            "top1_match": ours_top[0] == ref_top[0],
        })
    return {"reference": "hf/fla", "rows": rows}


def compare_llamacpp(model: RWKV7Model, model_dir: str, device: str,
                     server_url: str, prompts: list[str], max_tokens: int) -> dict:
    """Greedy token match against llama-server (temperature 0, no EOS stop)."""
    tok = RWKVTrieTokenizer(_vocab_path(model_dir))

    def server_completions(prompt_ids: list[int], n: int) -> list[int]:
        # OpenAI-compatible endpoint: prompt accepts a list of token ids;
        # logprobs=1 yields the generated token ids (greedy at temperature 0).
        body = json.dumps({
            "prompt": prompt_ids,
            "max_tokens": n,
            "temperature": 0.0,
            "logprobs": 1,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            server_url.rstrip("/") + "/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        lp = choice.get("logprobs")
        if lp is not None:
            return [t["id"] for t in lp.get("content", [])]
        # fallback: no logprobs -> return text for a text-level comparison
        return choice.get("text", "")

    rows = []
    for prompt in prompts:
        ids = [1] + tok.encode(prompt)
        ours = model.greedy_generate(
            torch.tensor([ids], device=device), max_tokens + 1)
        t0 = time.perf_counter()
        ref = server_completions(ids, max_tokens)
        dt = time.perf_counter() - t0
        n = min(len(ours), len(ref))
        match = sum(1 for i in range(n) if ours[i] == ref[i])
        first_div = next((i for i in range(n) if ours[i] != ref[i]), None)
        rows.append({
            "prompt": prompt[:60],
            "compared": n,
            "match": match,
            "match_rate": match / n if n else 0.0,
            "first_divergence": first_div,
            "ours_head": ours[:12],
            "ref_head": ref[:12],
        })
    return {"reference": "llama.cpp", "server_url": server_url, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ref", default="hf", choices=["hf", "llamacpp"])
    ap.add_argument("--server-url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    prompts = list(DEFAULT_PROMPTS)
    if args.prompt:
        prompts = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts += [ln.strip() for ln in f if ln.strip()]

    model = RWKV7Model.from_hf_dir(args.model_dir, device=args.device)
    if args.ref == "hf":
        result = compare_hf(model, args.model_dir, args.device, prompts, args.max_tokens, None)
    else:
        result = compare_llamacpp(model, args.model_dir, args.device,
                                  args.server_url, prompts, args.max_tokens)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
