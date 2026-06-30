#!/usr/bin/env bash
# Steady-state decode benchmark. Usage: tools/speed.sh [n_gen]
# Short n_gen (default 128) = short context = the engine's peak rate.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf}"
NGEN="${1:-128}"
PROMPT=$'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'
IDS=$(./build/ref_oracle "$MODEL" 1 "$PROMPT" 2>/dev/null | sed -n 's/^INPUT //p')
RAWIDS=1 ./build/cuda_infer "$MODEL" "$NGEN" $IDS >/dev/null
