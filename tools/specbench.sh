#!/usr/bin/env bash
# Speculative-decoding benchmark over a fixed prompt suite: cuda_infer (plain and its own SPEC=1)
# vs mega (SPEC=0 and SPEC=1). Checks that mega's speculative output is token-for-token its plain
# output (it must be: lossless by construction), and reports tok/s and tokens per step.
# Usage: tools/specbench.sh [n_gen]   env: MODEL, CUDA_INFER, MEGA, ORACLE
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf}"
CUDA_INFER="${CUDA_INFER:-./build/cuda_infer}"; MEGA="${MEGA:-./build/mega}"; ORACLE="${ORACLE:-./build/ref_oracle}"
N="${1:-256}"
T="${TMPDIR:-/tmp}/minfer-spec"; mkdir -p "$T"
prompts=(
  "Write a short story about a lighthouse keeper."
  "Write a Python function that checks whether a string is a palindrome, with tests."
  "List the first 30 prime numbers, one per line."
  "Explain how a transformer language model works to a high-school student."
  "Rewrite this list as a JSON array of objects with name and age: Alice 31, Bob 25, Carol 47, Dave 19, Eve 52, Frank 38."
)
rate() { sed -n 's/.*= \([0-9.]*\) tok\/s.*/\1/p' "$1" | head -1; }
printf "%-36s %9s %9s %9s %9s %7s %s\n" prompt cuda cuda-spec mega mega-spec tok/stp check
for i in "${!prompts[@]}"; do
  f="$T/p$i.ids"
  for _ in 1 2 3; do
    [ -s "$f" ] && break
    "$ORACLE" "$MODEL" 1 "$(printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "${prompts[$i]}")" 2>/dev/null \
      | sed -n 's/^INPUT //p' > "$f"
  done
  ids=$(cat "$f")
  RAWIDS=1 "$CUDA_INFER" "$MODEL" "$N" $ids >/dev/null 2>"$T/c0.err"
  RAWIDS=1 SPEC=1 "$CUDA_INFER" "$MODEL" "$N" $ids >/dev/null 2>"$T/c1.err"
  a=$(RAWIDS=1 SPEC=0 "$MEGA" "$MODEL" "$N" $ids 2>"$T/m0.err")
  b=$(RAWIDS=1 SPEC=1 "$MEGA" "$MODEL" "$N" $ids 2>"$T/m1.err")
  tps=$(sed -n 's/.*steps, \([0-9.]*\) tokens\/step.*/\1/p' "$T/m1.err")
  chk=$([ "$a" = "$b" ] && echo "identical" || echo "DIFFERENT")
  printf "%-36.36s %9s %9s %9s %9s %7s %s\n" "${prompts[$i]}" "$(rate $T/c0.err)" "$(rate $T/c1.err)" "$(rate $T/m0.err)" "$(rate $T/m1.err)" "$tps" "$chk"
done
