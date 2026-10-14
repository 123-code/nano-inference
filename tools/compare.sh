#!/usr/bin/env bash
# M1 correctness gate: minfer cpu_infer vs llama.cpp oracle, in token-ID space.
# Usage: tools/compare.sh [model.gguf] [n_gen] [prompt]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf}"
NGEN="${2:-12}"
PROMPT="${3:-$'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'}"

out=$(./build/ref_oracle "$MODEL" "$NGEN" "$PROMPT" 2>/dev/null)
IDS=$(echo "$out" | awk '/^INPUT/{$1="";print}')
REF=$(echo "$out" | awk '/^OUTPUT/{$1="";print}' | xargs)
GOT=$(OMP_NUM_THREADS=6 ./build/cpu_infer "$MODEL" "$NGEN" $IDS 2>/dev/null | xargs)

echo "ref:    $REF"
echo "minfer: $GOT"
if [ "$REF" = "$GOT" ]; then echo "PASS: greedy token sequences match"; exit 0
else echo "FAIL: sequences differ"; exit 1; fi
