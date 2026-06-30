#!/usr/bin/env bash
# Profile the dense q4 GEMV (gemv_dp4a) to find its limiter: compute vs memory.
# Needs sudo (ncu requires elevated GPU perf counters). Run:  sudo bash tools/ncu_gemv.sh
# Read the SpeedOfLight section: "Compute (SM) [%]" vs "Memory [%]" — whichever is
# higher (and near 100) is the bound. Occupancy section shows if we're warp-starved.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf
NCU=/usr/local/cuda/bin/ncu
PROMPT=$'<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'
IDS=$(./build/ref_oracle "$MODEL" 1 "$PROMPT" 2>/dev/null | sed -n 's/^INPUT //p')

# Skip prefill + a couple decode steps to land on steady-state decode launches,
# then sample 8 gemv_dp4a instances (covers the different matmul shapes).
"$NCU" --kernel-name regex:gemv_dp4a --launch-skip 3500 --launch-count 8 \
  --section SpeedOfLight --section Occupancy --section WarpStateStats \
  ./build/cuda_infer "$MODEL" 16 $IDS
