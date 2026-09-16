#!/usr/bin/env bash
# One-shot setup + acceptance on V100_222 (GPU0, 32G V100).
# Idempotent: skips already-done steps.
set -uo pipefail
GPU=${GPU:-0}
HF_DIR=${HF_DIR:-/data/run/g1j-2.9b-hf}
GGUF=${GGUF:-/data/run/g1j-2.9b-f16.gguf}
PY=${PY:-$HOME/siphon-rwkv7/bin/python}
LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
cd $HOME/siphon/rwkv7-engine

echo "############ 1. torch ############"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'archs',torch.cuda.get_arch_list())" || {
  echo "torch missing, installing...";
  $PY -m pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124;
  $PY -c "import torch;print('torch',torch.__version__)";
}

echo "############ 2. siphon _C + wkv .so ############"
# build the siphon C loader (editable, no deps) if not importable
$PY -c "import siphon" 2>/dev/null || ( cd $HOME/siphon && $PY -m pip install -q -e . --no-deps )
rm -f rwkv7_engine/_wkv7.so
CUDA_VISIBLE_DEVICES=$GPU $PY -c "from rwkv7_engine.wkv import wkv7; print('wkv .so built')"

echo "############ 3. kernel unit test ############"
CUDA_VISIBLE_DEVICES=$GPU $PY tests/test_wkv.py

echo "############ 4. llama.cpp build ############"
ls $LLAMA_BIN/llama-bench $LLAMA_BIN/llama-server >/dev/null 2>&1 || {
  cd $HOME/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCUDA_ARCH=70 >/dev/null 2>&1
  cmake --build build --config Release -j 32 2>&1 | tail -3
}
ls -la $LLAMA_BIN/ | grep -E "llama-bench|llama-server"

echo "############ 5. correctness: siphon vs llama.cpp ############"
# start llama-server on the f16 gguf
pkill -f "llama-server.*g1j" 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=$GPU $LLAMA_BIN/llama-server -m "$GGUF" -ngl 99 \
  --port 8080 --no-web-search > /tmp/llama_server.log 2>&1 &
# wait for server
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; then echo "llama-server up"; break; fi
  sleep 2
done
CUDA_VISIBLE_DEVICES=$GPU $PY -m rwkv7_engine.compare --model-dir "$HF_DIR" --device cuda:0 \
  --ref llamacpp --server-url http://127.0.0.1:8080 --max-tokens 64 \
  --out /tmp/correctness_llamacpp.json
echo "--- llama.cpp correctness result ---"
$PY -c "
import json
d=json.load(open('/tmp/correctness_llamacpp.json'))
for r in d['rows']:
    print(f\"  prompt={r['prompt'][:40]!r} match={r['match']}/{r['compared']} ({r['match_rate']:.2%}) first_div={r['first_divergence']}\")
"

echo "############ 6. correctness: siphon vs HF/FLA (logits) ############"
CUDA_VISIBLE_DEVICES=$GPU $PY -m rwkv7_engine.compare --model-dir "$HF_DIR" --device cuda:0 \
  --ref hf --max-tokens 32 --out /tmp/correctness_hf.json
echo "--- HF/FLA correctness result ---"
$PY -c "
import json
d=json.load(open('/tmp/correctness_hf.json'))
for r in d['rows']:
    print(f\"  prompt={r['prompt'][:40]!r} cos={r['cosine']:.5f} max_abs={r['max_abs_diff']:.3f} top1_match={r['top1_match']}\")
"

echo "############ 7. performance benchmark ############"
bash bench_compare.sh

echo "############ ACCEPTANCE DONE ############"
echo "correctness (llama.cpp): /tmp/correctness_llamacpp.json"
echo "correctness (HF/FLA):    /tmp/correctness_hf.json"
echo "performance:             /tmp/bench_compare.json"
