#!/usr/bin/env bash
# End-to-end RWKV7 benchmark: Siphon engine vs llama.cpp, same V100, same fp16 weights.
#
#   1. cold load  — both engines measured "process spawn -> weights ready"
#                   (siphon: spawn -> from_hf_dir returned + cuda sync;
#                    llama.cpp: spawn -> the server logs "model loaded").
#                   The page cache is evicted (fadvise DONTNEED, no root needed)
#                   before every run, and the two engines alternate so disk drift
#                   cannot favour either side.
#   2. prefill    — tok/s over 2048 tokens
#   3. decode     — tok/s over 128 greedy tokens
#
# Usage (V100_222 GPU0):  bash bench_compare.sh
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
GPU=${GPU:-0}
HF_DIR=${HF_DIR:-/data/run/g1j-2.9b-hf}
GGUF=${GGUF:-/data/run/g1j-2.9b-f16.gguf}
PY=${PY:-$HOME/siphon-rwkv7/bin/python}
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
ENGINE_DIR=${ENGINE_DIR:-$HOME/siphon/rwkv7-engine}
PREFILL=${PREFILL:-2048}
DECODE=${DECODE:-128}
ROUNDS=${ROUNDS:-3}
OUT=${OUT:-/tmp/bench_compare.json}
PORT=${PORT:-8086}
cd "$ENGINE_DIR" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$ENGINE_DIR

fadvise_drop() {  # evict a file's pages from the page cache (no root needed)
  $PY - "$1" <<'PYEOF'
import os, sys
f = open(sys.argv[1], 'rb')
os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
f.close()
PYEOF
}

# The probe timestamps itself in epoch seconds so the harness can subtract the
# spawn instant exactly.  Timing the shell's *post-exit* clock instead would add
# the process teardown (freeing ~6 GB of device memory, ~0.4 s here), which is
# not part of "how long until the model is ready".
cat > /tmp/siphon_cold_probe.py <<'PYEOF'
import sys, time, torch
T0 = time.perf_counter()
from rwkv7_engine.model import RWKV7Model
m = RWKV7Model.from_hf_dir(sys.argv[1], device='cuda:0', loader='siphon')
torch.cuda.synchronize()
print("READY %.6f %.3f" % (time.time(), time.perf_counter() - T0))
PYEOF

echo "### GPU: $(nvidia-smi -i $GPU --query-gpu=name,memory.free --format=csv,noheader)"
echo "### siphon model: $HF_DIR"
echo "### llama.cpp gguf: $GGUF"
echo "### rounds: $ROUNDS"

# ------------------------------------------------------------------ 1. cold load
SIPHON_COLD=(); LLAMA_COLD=(); LLAMA_INNER=()
for r in $(seq 1 "$ROUNDS"); do
  fadvise_drop "$HF_DIR/model.safetensors"
  t0=$(date +%s.%N)
  out=$($PY /tmp/siphon_cold_probe.py "$HF_DIR" 2>/tmp/siphon_cold.err)
  rc=$?
  t1=$(date +%s.%N)
  if [ $rc -ne 0 ]; then
    echo "round$r siphon     cold-load FAILED: $(tail -1 /tmp/siphon_cold.err)"
  else
    w=$(echo "$(awk '{print $2}' <<<"$out") - $t0" | bc); SIPHON_COLD+=("$w")
    echo "round$r siphon     cold-load (spawn -> ready):        $w   [$(awk '{print $3}' <<<"$out")s engine-internal, $(echo "$t1 - $t0" | bc)s incl. teardown]"
  fi

  fadvise_drop "$GGUF"
  rm -f /tmp/llama_cold_srv.log
  t0=$(date +%s.%N)
  stdbuf -oL setsid $LLAMA_BIN/llama-server -m "$GGUF" -ngl 99 \
    --port "$PORT" --host 127.0.0.1 >/tmp/llama_cold_srv.log 2>&1 &
  SRV=$!
  ok=0
  for _ in $(seq 1 3000); do
    grep -qa "model loaded" /tmp/llama_cold_srv.log 2>/dev/null && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 0.1
  done
  t1=$(date +%s.%N)
  if [ $ok -eq 1 ]; then
    w=$(echo "$t1 - $t0" | bc); LLAMA_COLD+=("$w")
    inner=$(grep -a "model loaded" /tmp/llama_cold_srv.log | head -1 | awk '{print $1}' \
            | awk -F. '{printf "%.3f", $1*60 + $2 + $3/1e3 + $4/1e6}')
    LLAMA_INNER+=("$inner")
    echo "round$r llama.cpp  cold-load (spawn -> model loaded): $w   [${inner}s engine-internal]"
  else
    echo "round$r llama.cpp  cold-load FAILED: $(tail -1 /tmp/llama_cold_srv.log)"
  fi
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2
done

# ------------------------------------------------------------------ 2/3. throughput
CUDA_VISIBLE_DEVICES=$GPU SIPHON_MAX_FREE_MEM_USAGE=0.6 $PY -m rwkv7_engine.bench \
  --model-dir "$HF_DIR" --device cuda:0 --loader siphon \
  --prefill-tokens $PREFILL --decode-tokens $DECODE --warmup 3 \
  --skip-load --out /tmp/siphon_bench.json >/dev/null 2>&1
SIPHON_PREFILL=$($PY -c "import json;print(json.load(open('/tmp/siphon_bench.json'))['prefill']['tok_per_s'])")
SIPHON_DECODE=$($PY -c "import json;print(json.load(open('/tmp/siphon_bench.json'))['decode']['tok_per_s'])")
echo "siphon    prefill: ${SIPHON_PREFILL} tok/s   decode: ${SIPHON_DECODE} tok/s"

CUDA_VISIBLE_DEVICES=$GPU $LLAMA_BIN/llama-bench -m "$GGUF" -ngl 99 \
  -n $DECODE -p $PREFILL > /tmp/llama_bench_raw.txt 2>&1
tail -4 /tmp/llama_bench_raw.txt
LLAMA_PREFILL=$(grep -E "pp${PREFILL}" /tmp/llama_bench_raw.txt | grep -oE "[0-9]+(\.[0-9]+)? *± *[0-9.]+" | head -1 | awk '{print $1}')
LLAMA_DECODE=$(grep -E "tg${DECODE}" /tmp/llama_bench_raw.txt | grep -oE "[0-9]+(\.[0-9]+)? *± *[0-9.]+" | head -1 | awk '{print $1}')
echo "llama.cpp prefill: ${LLAMA_PREFILL} tok/s   decode: ${LLAMA_DECODE} tok/s"

# ------------------------------------------------------------------ summary
$PY - "$OUT" "${SIPHON_COLD[*]}" "${LLAMA_COLD[*]}" "${LLAMA_INNER[*]}" \
      "$SIPHON_PREFILL" "$SIPHON_DECODE" "$LLAMA_PREFILL" "$LLAMA_DECODE" <<'PYEOF'
import json, statistics, sys
out, sc, lc, li, sp, sd, lp, ld = sys.argv[1:9]
def vals(s):
    return [float(x) for x in s.split() if x]
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
s_cold, l_cold, l_inner = vals(sc), vals(lc), vals(li)
med = lambda v: round(statistics.median(v), 3) if v else None
r = {
    "siphon":   {"cold_load_s": med(s_cold), "cold_load_runs": s_cold,
                 "prefill_tok_s": f(sp), "decode_tok_s": f(sd)},
    "llamacpp": {"cold_load_s": med(l_cold), "cold_load_runs": l_cold,
                 "cold_load_internal_s": med(l_inner), "cold_load_internal_runs": l_inner,
                 "prefill_tok_s": f(lp), "decode_tok_s": f(ld)},
}
s, l = r["siphon"], r["llamacpp"]
if s["cold_load_s"] and l["cold_load_s"] and s["prefill_tok_s"] and s["decode_tok_s"] \
        and l["prefill_tok_s"] and l["decode_tok_s"]:
    r["verdict"] = {
        "cold_load": "siphon" if s["cold_load_s"] <= l["cold_load_s"] else "llama.cpp",
        "prefill":   "siphon" if s["prefill_tok_s"] >= l["prefill_tok_s"] else "llama.cpp",
        "decode":    "siphon" if s["decode_tok_s"] >= l["decode_tok_s"] else "llama.cpp",
        "ratio": {
            "cold_load": round(l["cold_load_s"] / s["cold_load_s"], 3),
            "prefill":   round(s["prefill_tok_s"] / l["prefill_tok_s"], 3),
            "decode":    round(s["decode_tok_s"] / l["decode_tok_s"], 3),
        },
        "siphon_passes_all": (s["cold_load_s"] <= l["cold_load_s"]
                              and s["prefill_tok_s"] >= l["prefill_tok_s"]
                              and s["decode_tok_s"] >= l["decode_tok_s"]),
    }
print(json.dumps(r, indent=2))
open(out, "w").write(json.dumps(r, indent=2) + "\n")
PYEOF
echo "### summary written to $OUT"
