"""Official per-token reference (native.py attn_step/ffn_step) per-position logits.

Loads the HF model, runs the official sequential per-token loop, and records the
logits after every token. This is the ground-truth RWKV7 formula.

Usage:
  python -m rwkv7_engine.official_ref --model-dir D --prompt "..." --out /tmp/official.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import torch
import torch.nn.functional as F


def _load_native(model_dir: str):
    path = os.path.join(model_dir, "native.py")
    spec = importlib.util.spec_from_file_location("rwkv7_native_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prompt_ids(model_dir: str, prompt: str) -> list[int]:
    from .tokenizer import RWKVTrieTokenizer
    tok = RWKVTrieTokenizer(os.path.join(model_dir, "rwkv_vocab_v20230424.txt"))
    return [1] + tok.encode(prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    dev = args.device

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=torch.float16).to(dev)
    model.eval()
    native = _load_native(args.model_dir)

    base = model.model
    ids = _prompt_ids(args.model_dir, args.prompt)
    state, xpa, xpf, v_first = native._init_state(model, dev, base.embeddings.weight.dtype)

    all_logits = []
    x = None
    with torch.no_grad():
        for t in range(len(ids)):
            x = F.embedding(torch.tensor([[ids[t]]], device=dev), base.embeddings.weight).reshape(-1)
            x, state, xpa, xpf, v_first = native._step_token(model, x, state, xpa, xpf, v_first)
            x = base.norm(x)
            logits = F.linear(x, model.lm_head.weight)
            all_logits.append(logits.float().cpu())

    torch.save({"ids": ids, "all_logits": torch.stack(all_logits, dim=0)}, args.out)
    print(f"saved official per-token logits to {args.out} "
          f"(last top5: {all_logits[-1].topk(5).indices.tolist()})")


if __name__ == "__main__":
    main()
