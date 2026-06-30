#!/usr/bin/env bash
# Run minfer end-to-end from a plain prompt and stream the model's TEXT live.
# Usage: tools/run.sh "your prompt"  [n_gen]
#   env: SPEC=1 (n-gram spec decode), LMHEAD=0 (byte-exact fp32), RAWIDS=1 (token IDs)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf}"
USER_PROMPT="${1:-What is the capital of France?}"
NGEN="${2:-128}"

# Wrap in the Qwen2.5 chat template, then tokenize via the llama.cpp oracle.
PROMPT=$'<|im_start|>user\n'"$USER_PROMPT"$'<|im_end|>\n<|im_start|>assistant\n'
IDS=$(./build/ref_oracle "$MODEL" 1 "$PROMPT" 2>/dev/null | sed -n 's/^INPUT //p')
[ -z "$IDS" ] && { echo "tokenization failed (empty IDS) — retry (NvMap pressure?)" >&2; exit 1; }

printf '\033[2m%s\033[0m\n' "$USER_PROMPT"   # echo prompt dimmed
./build/cuda_infer "$MODEL" "$NGEN" $IDS
