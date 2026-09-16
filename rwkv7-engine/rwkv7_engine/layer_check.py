"""Per-layer divergence localisation.

Captures the post-FFN residual stream after every decoder layer from both the
HF reference (via forward hooks) and the Siphon RWKV7Model (via an in-forward
capture list), then reports the first layer where they diverge.

Usage:
  python -m rwkv7_engine.layer_check --model-dir D --prompt "..."
"""
from __future__ import annotations

import argparse
import os

import torch


def _prompt_ids(model_dir: str, prompt: str) -> list[int]:
    from .tokenizer import RWKVTrieTokenizer
    tok = RWKVTrieTokenizer(os.path.join(model_dir, "rwkv_vocab_v20230424.txt"))
    return [1] + tok.encode(prompt)


def capture_ref(model_dir: str, device: str, ids: list[int]) -> list[torch.Tensor]:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    model.eval()
    layers = model.model.layers
    captured: list[torch.Tensor] = []
    handles = []
    for i, layer in enumerate(layers):
        def hook(mod, inp, out, i=i):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            captured.append(hs[0].float().cpu())
        handles.append(layer.register_forward_hook(hook))
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        model(x, use_cache=False)
    for h in handles:
        h.remove()
    return captured


def capture_ours(model_dir: str, device: str, ids: list[int]) -> list[torch.Tensor]:
    from .model import RWKV7Model
    model = RWKV7Model.from_hf_dir(model_dir, device=device, loader="safetensors")
    x = torch.tensor([ids], device=device)
    captured: list[torch.Tensor] = []
    logits, _ = model.forward(x, model.init_state(), capture=captured)
    return captured


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)
    ids = _prompt_ids(args.model_dir, args.prompt)
    print(f"prompt ids: {ids}")

    ref = capture_ref(args.model_dir, args.device, ids)
    torch.cuda.empty_cache()
    ours = capture_ours(args.model_dir, args.device, ids)

    n = min(len(ref), len(ours))
    print(f"layers ref={len(ref)} ours={len(ours)}")
    for i in range(n):
        r, o = ref[i], ours[i]
        t = min(r.shape[1], o.shape[1])
        r, o = r[:, :t], o[:, :t]
        diff = (r - o).abs()
        cos = torch.nn.functional.cosine_similarity(r.flatten(1), o.flatten(1)).item()
        print(f"L{i:2d} max_abs={diff.max().item():10.4f} "
              f"mean_abs={diff.mean().item():9.5f} cosine={cos:.6f} "
              f"|r|={r.abs().mean().item():.3f} |o|={o.abs().mean().item():.3f}")


if __name__ == "__main__":
    main()
