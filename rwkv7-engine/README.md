# rwkv7-engine — RWKV7 (G1j) adaptation for Siphon

A hand-written RWKV7 inference engine that closes the loop the other two
trees study from each end: **Siphon loads the weights**, and this tree
**proves the forward pass is numerically right and fast**.

Target: `RWKV7-G1j-2.9B` (32 layers, C=2560, 40 heads × D=64,
vocab 65536), fp16 weights / fp32 Wkv state, on a V100 (sm_70).

## What it does

1. **Load** — `RWKV7Model.from_hf_dir` reads the HF-format checkpoint
   (config.json + safetensors) through the Siphon loader
   (`from siphon import safe_open`) straight to the GPU.
2. **Forward** — a dependency-free forward pass (no transformers, no FLA):
   token-shift + 6× addcmul + r/k/v/o projections + the four low-rank
   adapters (decay `w`, iclr `a`, value-residual `v`, output gate `g`) +
   the Wkv recurrence + group-norm/gate correction, then the channel mix
   (key → sqrelu → value). The Wkv recurrence is a custom CUDA kernel
   (`wkv_kernel.cu`) with fp16 I/O and fp32 state — the V100-friendly
   "fp32io16" mode.
3. **Verify** — `compare.py` checks logits against the HF/FLA reference
   and greedy tokens against llama.cpp; `tests/test_wkv.py` checks the
   kernel against a pure-torch fp32 recurrence.
4. **Bench** — `bench.py` measures cold load, prefill tok/s and decode
   tok/s; `bench_compare.sh` runs the head-to-head against llama.cpp with the
   page cache evicted before every round.  Results and the full breakdown:
   [`docs/rwkv7-vs-llamacpp.md`](docs/rwkv7-vs-llamacpp.md) — cold load at
   parity (13.10 s vs 13.14 s, both disk-bound), prefill **1.66×**, decode
   **1.007×**.

## The Wkv recurrence (per sequence, per head)

```
state = exp(w_t) * state + (a_t · state) * b_t + k_t * v_t
o_t   = state · r_t
```
with `a = -kk`, `b = kk * a`, `kk = l2norm(k * k_k)`, and
`w = -0.6065306597126334 * sigmoid(w_lora(xw))`. `o_t` is a scalar per
head, replicated over the head's 64 slots (the group norm then sees a
constant vector per group — a property shared by the FLA and llama.cpp
implementations).

## Layout

```
rwkv7_engine/
  config.py        architecture config (G1j-2.9B defaults)
  model.py         RWKV7Model: siphon load + forward + greedy decode
  wkv.py           standalone .so build (nvcc) + ctypes bindings
  wkv_kernel.cu    the Wkv recurrence kernel (fp16 in, fp32 state)
  tokenizer.py     RWKV trie tokenizer (rwkv_vocab_v20230424.txt)
  bench.py         cold-load / prefill / decode benchmark (CLI)
  compare.py       correctness vs HF/FLA and llama.cpp (CLI)
tests/
  test_wkv.py      kernel vs pure-torch reference
docs/
  rwkv7-vs-llamacpp.md   the head-to-head report (three metrics + correctness)
setup_and_accept.sh  one-shot setup + acceptance on the V100
run_correctness.sh   greedy tokens vs llama.cpp, then logits vs HF/FLA
bench_compare.sh     cold load / prefill / decode, Siphon vs llama.cpp
```

## Usage (on the V100)

```bash
# build the kernel (once)
python -c "from rwkv7_engine.wkv import build_wkv_lib; build_wkv_lib()"

# kernel unit tests
python tests/test_wkv.py

# correctness vs the HF/FLA reference
python -m rwkv7_engine.compare --model-dir $G1J_HF --ref hf

# end-to-end bench
python -m rwkv7_engine.bench --model-dir $G1J_HF \
    --prefill-tokens 2048 --decode-tokens 128 --out bench.json
```

## Notes

- The kernel is built with a plain `nvcc` invocation (no torch
  dependency), so it works with any torch CUDA build.
- fp16 weights with fp32 state match the llama.cpp reference exactly in
  precision class (llama.cpp also keeps the Wkv state in fp32).
