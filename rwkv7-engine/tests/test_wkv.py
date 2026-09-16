"""Unit test: Wkv CUDA kernel vs a pure-torch fp32 reference.

Run on the server:  python -m pytest tests/ -q   (or  python tests/test_wkv.py)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rwkv7_engine.wkv import wkv7


def _torch_ref(r, w, k, v, a, b, s_in, T, H, D):
    """Serial recurrence in fp32 (the ground truth). State is [H, D, D]."""
    out = torch.empty(T, H, D, dtype=torch.float32, device=r.device)
    state = s_in.float().clone() if s_in is not None else torch.zeros(H, D, D, device=r.device)
    for t in range(T):
        rr = r[t].float().view(H, D)
        ww = w[t].float().view(H, D)
        kk = k[t].float().view(H, D)
        vv = v[t].float().view(H, D)
        aa = a[t].float().view(H, D)
        bb = b[t].float().view(H, D)
        sa = (aa.unsqueeze(1) * state).sum(-1)                    # [H, D] a . row_i
        state = (state * torch.exp(ww).unsqueeze(1)
                 + kk.unsqueeze(1) * vv.unsqueeze(2)
                 + sa.unsqueeze(2) * bb.unsqueeze(1))
        out[t] = (state * rr.unsqueeze(1)).sum(-1)                # [H, D] row_i . r
    return out, state


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_wkv_kernel_matches_torch():
    T, H, D = 37, 3, 64
    dev = "cuda"
    seed = 1234
    r = _rand(T, H * D, seed=seed).half().to(dev)
    w = (-_rand(T, H * D, seed=seed + 1).abs() * 0.6).half().to(dev)  # realistic negative decay
    k = _rand(T, H * D, seed=seed + 2).half().to(dev)
    v = _rand(T, H * D, seed=seed + 3).half().to(dev)
    a = _rand(T, H * D, seed=seed + 4).half().to(dev)
    b = _rand(T, H * D, seed=seed + 5).half().to(dev)

    # no initial state
    got = wkv7(r, w, k, v, a, b, None, None)
    ref, _ = _torch_ref(r, w, k, v, a, b, None, T, H, D)
    ref = ref.reshape(T, H * D)
    assert got.shape == ref.shape
    diff = (got - ref).abs()
    rel = diff / ref.abs().clamp_min(1.0)
    assert rel.max().item() < 5e-3, f"max rel diff {rel.max().item()}"

    # with initial state
    s0 = _rand(H, D, D, seed=seed + 6).float().to(dev)
    s_out = torch.empty(H, D, D, dtype=torch.float32, device=dev)
    got2 = wkv7(r, w, k, v, a, b, s0, s_out)
    ref2, s_final = _torch_ref(r, w, k, v, a, b, s0, T, H, D)
    ref2 = ref2.reshape(T, H * D)
    assert (got2 - ref2).abs().max().item() < 1e-3 * ref2.abs().max().item()
    assert (s_out - s_final).abs().max().item() < 1e-3 * s_final.abs().max().item()

    # single-token decode
    got3 = wkv7(r[:1], w[:1], k[:1], v[:1], a[:1], b[:1], s0, None)
    ref3, _ = _torch_ref(r[:1], w[:1], k[:1], v[:1], a[:1], b[:1], s0, 1, H, D)
    ref3 = ref3.reshape(1, H * D)
    assert (got3 - ref3).abs().max().item() < 1e-3 * ref3.abs().max().item()
    print("test_wkv_kernel_matches_torch OK")


def test_wkv_long_sequence_stability():
    """Longer sequence: check no blow-up and scale-aware agreement."""
    T, H, D = 512, 2, 64
    dev = "cuda"
    g = torch.Generator().manual_seed(99)
    r = torch.randn(T, H * D, generator=g).half().to(dev)
    w = (torch.randn(T, H * D, generator=g) * 0.2 - 1.0).half().to(dev)
    k = torch.randn(T, H * D, generator=g).half().to(dev)
    v = torch.randn(T, H * D, generator=g).half().to(dev)
    a = torch.nn.functional.normalize(torch.randn(T, H, D, generator=g), dim=-1).reshape(T, H * D).half().to(dev)
    b = (torch.nn.functional.normalize(torch.randn(T, H, D, generator=g), dim=-1).reshape(T, H * D) * 0.5).half().to(dev)
    got = wkv7(r, w, k, v, a, b, None, None)
    ref, _ = _torch_ref(r, w, k, v, a, b, None, T, H, D)
    ref = ref.reshape(T, H * D)
    finite = torch.isfinite(got).all().item()
    ref_finite = torch.isfinite(ref).all().item()
    assert finite == ref_finite, "kernel finite-ness disagrees with reference"
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err < 1e-3 * max(scale, 1.0), f"max abs diff {err} vs scale {scale}"
    print(f"test_wkv_long_sequence_stability OK (max abs {err:.2e} vs scale {scale:.2e}, finite={finite})")


if __name__ == "__main__":
    test_wkv_kernel_matches_torch()
    test_wkv_long_sequence_stability()
    print("all wkv tests passed")
