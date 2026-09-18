"""Decode profiling: GPU time vs wall time per token + top kernel/op breakdown."""
import sys, time, torch

sys.path.insert(0, ".")
from rwkv7_engine.model import RWKV7Model

HF_DIR = "/data/run/g1j-2.9b-hf"
torch.cuda.set_device(0)
t0 = time.perf_counter()
m = RWKV7Model.from_hf_dir(HF_DIR, device="cuda:0", loader="siphon")
print(f"load: {time.perf_counter()-t0:.2f}s")

ids = torch.tensor([[1, 6699, 51128, 4706, 44312, 4600]], device="cuda:0")
x1 = ids[:, -1:]

state = m.init_state()
logits, state = m.forward(ids, state)  # prefill (warmup)
for _ in range(3):
    logits, state = m.forward(x1, state)
torch.cuda.synchronize()

N = 64
ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
ev_s.record()
t_wall = time.perf_counter()
for _ in range(N):
    logits, state = m.forward(x1, state)
torch.cuda.synchronize()
wall = time.perf_counter() - t_wall
ev_e.record()
torch.cuda.synchronize()
gpu_ms = ev_s.elapsed_time(ev_e)
print(f"per-token: wall={wall/N*1000:.3f}ms  gpu={gpu_ms/N:.3f}ms  -> {'CPU-bound' if wall > gpu_ms*1.3 else 'GPU-bound'}")
print(f"decode: {N/wall:.2f} tok/s")

# --- profile: top CUDA kernels and CPU ops ---
from torch.profiler import profile, ProfilerActivity
state = m.init_state()
logits, state = m.forward(ids, state)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(16):
        logits, state = m.forward(x1, state)
    torch.cuda.synchronize()
print("\n=== top CUDA kernels (16 tokens) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
print("\n=== top CPU ops (16 tokens) ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
