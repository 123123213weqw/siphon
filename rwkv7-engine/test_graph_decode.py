"""Test CUDA-graph decode: correctness vs eager path + speed."""
import sys, time, torch

sys.path.insert(0, ".")
from rwkv7_engine.model import RWKV7Model

HF_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/run/g1j-2.9b-hf"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
torch.cuda.set_device(DEV)

m = RWKV7Model.from_hf_dir(HF_DIR, device=DEV, loader="siphon")
ids = torch.tensor([[1, 6699, 51128, 4706, 44312, 4600]], device=DEV)

# eager reference
eager = m.greedy_generate(ids, 48, use_graph=False)
print("eager :", eager[:24])

# graph path
t0 = time.perf_counter()
graph = m.greedy_generate(ids, 48, use_graph=True)
torch.cuda.synchronize(DEV)
print(f"graph : {graph[:24]}  (capture+gen {time.perf_counter()-t0:.2f}s)")

match = sum(1 for a, b in zip(eager, graph) if a == b)
print(f"token match: {match}/{min(len(eager), len(graph))}  "
      f"{'PASS' if match == min(len(eager), len(graph)) else 'FAIL'}")

# steady-state speed (graph already captured)
N = 128
t0 = time.perf_counter()
for _ in range(3):
    m.greedy_generate(ids, N, use_graph=True)
torch.cuda.synchronize(DEV)
wall = time.perf_counter() - t0
print(f"graph decode (incl prefill 6 tok): {3*N/wall:.1f} tok/s total")
