#!/usr/bin/env bash
# End-to-end RWKV7 benchmark: Siphon engine vs llama.cpp, same V100, same fp16 weights.
#
# Measures three metrics the acceptance requires:
#   1. cold load  (wall seconds to load all weights to GPU from a cold page cache)
#   2. prefill    (tok/s for a 2048-token prompt)
#   3. decode     (tok/s for greedy generation of 128 tokens)
#
# Usage (on V100_222, GPU0):
#   bash bench_compare.sh
set -uo pipefail

GPU=${GPU:-0}
HF_DIR=${HF_DIR:-/data/run/g1j-2.9b-hf}
GGUF=${GGUF:-/data/run/g1j-2.9b-f16.gguf}
PY=${PY:-$HOME/siphon-rwkv7/bin/python}
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
PREFILL=2048
DECODE=128
OUT=${OUT:-/tmp/bench_compare.json}
cd $HOME/siphon/rwkv7-engine

drop_caches() {
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
  sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
}

echo "### GPU: $(CUDA_VISIBLE_DEVICES=$GPU nvidia-smi -i $GPU --query-gpu=name,memory.free --format=csv,noheader)"
echo "### siphon model: $HF_DIR"
echo "### llama.cpp gguf: $GGUF"

# ---------------------------------------------------------------- siphon
# Cold load in a FRESH process (drop caches first so it is a real disk read).
drop_caches
SIPHON_LOAD=$(CUDA_VISIBLE_DEVICES=$GPU SIPHON_MAX_FREE_MEM_USAGE=0.6 $PY -c "
import time, torch
t0=time.perf_counter()
from rwkv7_engine.model import RWKV7Model
m=RWKV7Model.from_hf_dir('$HF_DIR', device='cuda:0', loader='siphon')
torch.cuda.synchronize()
print(f'{time.perf_counter()-t0:.3f}')
")
echo "siphon cold load: ${SIPHON_LOAD}s"

# prefill + decode (same process, warm)
CUDA_VISIBLE_DEVICES=$GPU SIPHON_MAX_FREE_MEM_USAGE=0.6 $PY -m rwkv7_engine.bench \
  --model-dir "$HF_DIR" --device cuda:0 --loader siphon \
  --prefill-tokens $PREFILL --decode-tokens $DECODE --warmup 3 \
  --skip-load --out /tmp/siphon_bench.json
SIPHON_PREFILL=$(python3 -c "import json;print(json.load(open('/tmp/siphon_bench.json'))['prefill']['tok_per_s'])")
SIPHON_DECODE=$(python3 -c "import json;print(json.load(open('/tmp/siphon_bench.json'))['decode']['tok_per_s'])")
echo "siphon prefill: ${SIPHON_PREFILL} tok/s   decode: ${SIPHON_DECODE} tok/s"

# ---------------------------------------------------------------- llama.cpp
# Cold load: capture llama.cpp's own reported model load time (fresh process).
drop_caches
LLAMA_LOAD=$(CUDA_VISIBLE_DEVICES=$GPU $LLAMA_BIN/llama-cli -m "$GGUF" -ngl 99 \
  -p "hi" -n 0 2>&1 | grep -oE "model load time = [0-9]+ ms" | grep -oE "[0-9]+" | head -1)
LLAMA_LOAD=$(awk "BEGIN{printf \"%.3f\", ${LLAMA_LOAD:-0}/1000}")
echo "llama.cpp cold load: ${LLAMA_LOAD}s"

# prefill + decode via llama-bench (pp 2048, tg 128); capture raw, parse robustly
CUDA_VISIBLE_DEVICES=$GPU $LLAMA_BIN/llama-bench -m "$GGUF" -ngl 99 \
  -n $DECODE -p $PREFILL > /tmp/llama_bench_raw.txt 2>&1
tail -4 /tmp/llama_bench_raw.txt
# last table row: ... | pp | tg |  -> take the last two numeric fields
LLAMA_ROW=$(grep -E "\| *[0-9]+(\.[0-9]+)? *\|" /tmp/llama_bench_raw.txt | tail -1)
LLAMA_PREFILL=$(echo "$LLAMA_ROW" | tr '|' '\n' | grep -E "^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$" | tail -2 | head -1 | tr -d ' ')
LLAMA_DECODE=$(echo "$LLAMA_ROW" | tr '|' '\n' | grep -E "^[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$" | tail -1 | tr -d ' ')
echo "llama.cpp prefill: ${LLAMA_PREFILL} tok/s   decode: ${LLAMA_DECODE} tok/s"

# ---------------------------------------------------------------- summary
python3 - "$SIPHON_LOAD" "$SIPHON_PREFILL" "$SIPHON_DECODE" \
         "$LLAMA_LOAD" "$LLAMA_PREFILL" "$LLAMA_DECODE" "$OUT" <<'PY'
import json, sys
sl, sp, sd, ll, lp, ld, out = sys.argv[1:8]
def f(x):
    try: return float(x)
    except: return None
r = {
    "siphon":  {"cold_load_s": f(sl), "prefill_tok_s": f(sp), "decode_tok_s": f(sd)},
    "llamacpp":{"cold_load_s": f(ll), "prefill_tok_s": f(lp), "decode_tok_s": f(ld)},
    "verdict": {
        "cold_load_ge":  (f(sl) is not None and f(ll) is not None and sl and f(ll) >= f(sl)),
        "prefill_ge":    (f(sp) is not None and f(lp) is not None and f(sp) >= f(lp)),
        "decode_ge":     (f(sd) is not None and f(ld) is not None and f(sd) >= f(ld)),
    },
}
# cold load: lower is better, so siphon wins if sl <= ll
r["verdict"]["cold_load_ge"] = (f(sl) is not None and f(ll) is not None and f(sl) <= f(ll))
print(json.dumps(r, indent=2))
open(out, "w").write(json.dumps(r, indent=2) + "\n")
PY
echo "### summary written to $OUT"
