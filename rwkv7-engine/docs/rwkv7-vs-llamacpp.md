# RWKV7-G1j-2.9B on Siphon vs llama.cpp — head-to-head on one V100

Three end-to-end metrics, one machine, one checkpoint, fp16 weights on both
sides.  Every number below is reproducible with `bench_compare.sh`.

## Result

| metric | siphon | llama.cpp | ratio | verdict |
| --- | --- | --- | --- | --- |
| cold load (spawn → weights ready) | **13.10 s** | 13.14 s | 1.003× | parity (both bound by the disk) |
| prefill 2048 tok | **7112 tok/s** | 4282 tok/s | **1.66×** | siphon |
| decode 128 tok (greedy) | **74.02 tok/s** | 73.51 tok/s | 1.007× | siphon |

Median of 3 alternating rounds; the page cache is evicted with
`posix_fadvise(DONTNEED)` before every run and the two engines alternate so a
drifting disk cannot favour either side.

## Test setting

| | |
| --- | --- |
| GPU | Tesla V100-PCIE-32GB (sm_70), 1 card |
| CPU / RAM | 88 threads, 125 GB |
| Checkpoint | `RWKV7-G1j-2.9B` fp16 — 32 layers, C=2560, 40 heads × D=64, inter 10240, vocab 65536 |
| Weights (HF safetensors) | 5.895 GB — read by Siphon |
| Weights (GGUF f16) | 5.932 GB — read by llama.cpp |
| Storage | PERC H730 SAS disk (`rotational=1`), ext4 — **0.52 GB/s sequential** |
| Siphon | 0.1.10, loader backend `URING`, direct I/O |
| llama.cpp | `llama-server` / `llama-bench`, `-ngl 99` (full offload), mmap |
| Prefill/decode | `rwkv7_engine.bench` vs `llama-bench -p 2048 -n 128` |

## Why cold load is a tie

The model file alone needs 5.9 GB / 0.52 GB/s ≈ **11.3 s** on this disk, and
nothing either engine does can go below that.  Measured sequentially:

| phase | siphon | llama.cpp |
| --- | --- | --- |
| process spawn → runtime ready | 0.03 s + 1.66 s (PyTorch import) | 0.15 s |
| runtime ready → weights on GPU | **11.42 s** (0.51 GB/s) | 12.99 s (0.46 GB/s, own log timestamp) |
| **spawn → ready** | **13.10 s** | **13.14 s** |

So Siphon's *loader* is 1.5 s faster than llama.cpp's (it streams at 98 % of
the device's sequential rate, where llama.cpp's mmap path lands at 90 %), and
its 1.7 s PyTorch import gives that advantage back.  The result is a dead heat
— on a disk that is 16× slower than the GPU.

Two measurement details worth keeping:

* The page cache must be evicted per run; otherwise both engines "load" in
  under a second and the comparison says nothing.
* The clock must be read *before* the loader process exits.  Tearing down a
  CUDA context that holds 6 GB costs ~0.4 s here, and a shell that reads the
  time after `wait` silently bills that to the load time.

### The loader setting that mattered

Siphon's automatic direct-I/O depth is 512.  That is the right answer for an
NVMe device and badly wrong for a spinning disk: the drive seeks between the
in-flight 8 MiB requests and loses ~25 % of its bandwidth (0.39 GB/s instead of
0.52 GB/s, i.e. 15.2 s instead of 11.4 s for this checkpoint).  Depth 16 sits at
the floor and was flat between 8 and 32.

`siphon/_impl.py` now probes `/sys/dev/block/*/queue/rotational` (walking up
from a partition node, which often has no `queue` directory of its own) and caps
the automatic depth at `DEFAULT_ROTATIONAL_IO_DEPTH = 16` for rotating media.
An explicit `io_depth` or `SIPHON_IO_DEPTH` still wins.  This is a
weight-loading change only — no effect on prefill or decode.

| | before | after |
| --- | --- | --- |
| engine phase | 15.2 s | 11.42 s |
| spawn → ready | 17.0–17.7 s | 13.10 s |

## Throughput

Decode is single-token, so every weight is read once per token: 5.3–5.4 GB per
token.  On a V100 with a measured 0.87 TB/s streaming ceiling that is a hard
6.0 ms/token = 166 tok/s; Siphon lands at 13.5 ms (74 tok/s), i.e. 45 % of the
ceiling, while llama.cpp's `tg128` is 13.6 ms.

The per-token budget (16-token profile, before the last three optimisations):

| stage | ms/token |
| --- | --- |
| `gemv_partial` (r/k/v/o/ffn/lm_head, fp16, split-K) | 5.69 |
| cuBLAS (unfused leftovers) | 2.96 |
| LoRA 4-gate batch (`lora_h1_partial` + `lora_gates_partial`) | 0.98 |
| Wkv recurrence (`wkv7_split_kernel`, one block per head) | 0.66 → 0.48 |
| `gemv_reduce`, layer-norm, `wkv7_post` | ~1.0 |

Decode improved 56.4 → 74.0 tok/s through: routing the projections to the
hand-written split-K GEMV, fusing the Wkv recurrence into one kernel, batching
the four LoRA gates into one launch, moving the activation function into the
reduction kernel, tuning `SIPHON_LORA_PHASES=128` / `SIPHON_LORA_SPLITR=16`,
and splitting the Wkv state update across 4 sub-groups per head with warp
shuffles.

Prefill is 1.66× ahead because it is a GEMM workload: 2048 tokens in one
batched forward, dominated by cuBLAS and the batched Wkv scan, where
llama.cpp's RWKV implementation stays closer to a token loop.

## Correctness

* **vs llama.cpp (same fp16 weights, greedy)**: 4/4 prompts, 64/64 tokens each,
  no first divergence.  `rwkv7_engine/run_correctness.sh` Part A.
* **vs the official native RWKV7 implementation** (`RWKV7_NATIVE_MODEL=1`,
  logits per token): 4/4 prompts, cosine 1.00000, max |Δ| 0.07–0.20, top-1
  token identical.  Part B.
* **internal**: fused decode / unfused decode / cuBLAS-only paths all produce
  the identical token stream; the Wkv kernel matches a pure-fp32 torch
  recurrence to 3.8e-6; `test_gemv.py` covers 24 shape combinations of the GEMV
  kernel (rel. err < 5e-4); `test_graph_decode.py` replays the captured CUDA
  graph 48/48.

The one bug worth recording: `torch._foreach_addcmul` (no trailing underscore)
is functional — the token-shift deltas were computed and thrown away, and the
engine produced fluent-looking but wrong tokens.  In-place
`torch._foreach_addcmul_` fixed it.

## Reproducing

```bash
# on the GPU box, from rwkv7-engine/
bash bench_compare.sh          # cold load x3 (alternating) + prefill + decode
bash run_correctness.sh        # Part A vs llama.cpp, Part B vs native RWKV7
python tests/test_wkv.py
python test_gemv.py
```

`bench_compare.sh` writes `bench_compare.json` with the per-round timings and
the three verdicts.
