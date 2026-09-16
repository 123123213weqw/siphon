#!/usr/bin/env bash
# Correctness: siphon RWKV7 engine (G1j-2.9B fp16) vs llama.cpp + HF/FLA.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
GPU=${GPU:-0}
HF_DIR=${HF_DIR:-/data/run/g1j-2.9b-hf}
GGUF=${GGUF:-/data/run/g1j-2.9b-f16.gguf}
PY=${PY:-$HOME/siphon-rwkv7/bin/python}
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
PORT=${PORT:-8081}
cd $HOME/siphon/rwkv7-engine

echo "############ A. siphon vs llama.cpp (greedy tokens) ############"
pkill -f "g1j-2.9b-f16.gguf" 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=$GPU $LLAMA_BIN/llama-server -m "$GGUF" -ngl 99 \
  --port $PORT --host 127.0.0.1 > /tmp/srv.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 90); do
  if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then up=1; break; fi
  sleep 2
done
echo "server up: $up  (health=$(curl -s http://127.0.0.1:$PORT/health 2>/dev/null))"

CUDA_VISIBLE_DEVICES=$GPU $PY -m rwkv7_engine.compare --model-dir "$HF_DIR" --device cuda:0 \
  --ref llamacpp --server-url "http://127.0.0.1:$PORT" --max-tokens 64 \
  --out /tmp/correctness_llamacpp.json
kill $SRV 2>/dev/null
echo "--- llama.cpp token match ---"
$PY - <<'PYEOF'
import json
d=json.load(open('/tmp/correctness_llamacpp.json'))
for r in d['rows']:
    print(f"  {r['prompt'][:44]!r:48} match={r['match']}/{r['compared']} ({r['match_rate']:.1%}) first_div={r['first_divergence']}")
PYEOF

echo ""
echo "############ B. siphon vs HF/FLA (per-token logits) ############"
CUDA_VISIBLE_DEVICES=$GPU $PY -m rwkv7_engine.compare --model-dir "$HF_DIR" --device cuda:0 \
  --ref hf --max-tokens 32 --out /tmp/correctness_hf.json 2>/tmp/hf_err.log
echo "--- HF/FLA logit stats ---"
$PY - <<'PYEOF'
import json
try:
    d=json.load(open('/tmp/correctness_hf.json'))
    for r in d['rows']:
        print(f"  {r['prompt'][:40]!r:44} cos={r['cosine']:.5f} max_abs={r['max_abs_diff']:.3f} top1={r['top1_match']}")
except Exception as e:
    print("  HF/FLA compare failed:", e)
    import subprocess
    print(subprocess.run(['tail','-6','/tmp/hf_err.log'],capture_output=True,text=True).stdout)
PYEOF
echo "############ CORRECTNESS DONE ############"
