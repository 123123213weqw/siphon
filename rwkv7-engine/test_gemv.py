"""GEMV kernel unit test: vs torch.matmul on all RWKV7 shapes."""
import sys, torch

sys.path.insert(0, ".")
from rwkv7_engine.wkv import gemv16

torch.manual_seed(0)
dev = "cuda:0"
shapes = [
    (2560, 2560),    # r/k/v/o (g1j)
    (2048, 2048),    # 1.5b attn
    (2560, 10240),   # ffn_key
    (10240, 2560),   # ffn_value
    (8192, 2048),    # 1.5b ffn
    (2560, 65536),   # lm_head
    (2560, 64),      # w_l1 (rank 64)
    (2560, 96),      # a_l1
    (2560, 320),     # v_l1
    (64, 2560),      # w_l0^T small-K
    (96, 2560),      # a_l0^T
    (320, 2560),     # v_l0^T
    (256, 512),      # arbitrary
]
ok = True
for K, N in shapes:
    W = (torch.randn(K, N, device=dev) * 0.05).half()
    x = (torch.randn(K, device=dev) * 0.5).half()
    for out32 in (False, True):
        y = gemv16(W, x, out_f32=out32)
        ref = (x.float() @ W.float())
        if out32:
            err = (y.float() - ref).abs().max().item()
            denom = ref.abs().max().item() + 1e-9
        else:
            err = (y.float() - ref.half().float()).abs().max().item()
            denom = ref.abs().max().item() + 1e-9
        status = "OK" if err / denom < 2e-3 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"K={K:6d} N={N:6d} out32={out32}  max_rel={err/denom:.2e} {status}")
print("ALL PASS" if ok else "FAILURES")
