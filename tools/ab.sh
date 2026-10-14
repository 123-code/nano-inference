#!/usr/bin/env bash
# A/B harness: cuda_infer (baseline) vs mega on a fixed prompt suite.
#   speed:   free-running greedy decode, tok/s of each engine, first divergence + mega's margin there
#   teacher: feed mega cuda_infer's own sequence; top-1 agreement per position (margin on misses)
# Usage: tools/ab.sh [n_gen]   env: MODEL, CUDA_INFER, MEGA, ORACLE (binary paths), MODE=speed|teacher|both
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf}"
CUDA_INFER="${CUDA_INFER:-./build/cuda_infer}"; MEGA="${MEGA:-./build/mega}"; ORACLE="${ORACLE:-./build/ref_oracle}"
N="${1:-256}"; MODE="${MODE:-both}"
T="${TMPDIR:-/tmp}/minfer-ab"; mkdir -p "$T"
prompts=(
  "Write a short story about a lighthouse keeper."
  "Write a Python function that checks whether a string is a palindrome, with tests."
  "List the first 30 prime numbers, one per line."
  "Explain how a transformer language model works to a high-school student."
  "What is the capital of France? Answer in one word, then give three facts about it."
)
ids_for() {   # tokenize once (cached); the llama.cpp oracle occasionally returns nothing, so retry
  local f="$T/p$1.ids"
  for _ in 1 2 3; do
    [ -s "$f" ] && break
    "$ORACLE" "$MODEL" 1 "$(printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "$2")" 2>/dev/null \
      | sed -n 's/^INPUT //p' > "$f"
  done
  [ -s "$f" ] || { echo "tokenization failed for prompt $1" >&2; return 1; }
  cat "$f"
}
agree=0; total=0
for i in "${!prompts[@]}"; do
  ids=$(ids_for "$i" "${prompts[$i]}") || continue
  ref=$(RAWIDS=1 "$CUDA_INFER" "$MODEL" "$N" $ids 2>"$T/ref.err")
  if [ "$MODE" != teacher ]; then
    got=$(RAWIDS=1 MARGINS=1 "$MEGA" "$MODEL" "$N" $ids 2>"$T/mega.err")
    python3 -B -I - "$ref" "$got" "$T" <<'PY'
import re, sys
a, b, t = sys.argv[1].split(), sys.argv[2].split(), sys.argv[3]
rt = re.search(r'= ([\d.]+) tok/s', open(t + '/ref.err').read()).group(1)
me = open(t + '/mega.err').read()
mt = re.search(r'decode: .*= ([\d.]+) tok/s', me).group(1)
marg = dict((int(x.split(':')[0]), float(x.split(':')[2])) for x in me.strip().split('\n')[-1].split())
n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
tail = f"MATCH all {len(a)}" if n is None else f"diverge at {n}/{len(a)} (mega margin {marg.get(n, float('nan')):.4f})"
print(f"  speed   cuda_infer {rt:>6} tok/s   mega {mt:>6} tok/s   {tail}")
PY
  fi
  if [ "$MODE" != speed ]; then
    out=$(TEACHER=$(echo $ids | wc -w) "$MEGA" "$MODEL" 1 $ids $ref 2>&1 >/dev/null)
    misses=$(echo "$out" | grep -c '^  pos ' || true)
    line=$(echo "$out" | grep agreement); echo "  teacher $line $(echo "$out" | grep '^  pos ' | sed 's/.*margin \([0-9.]*\)).*/\1/' | tr '\n' ' ' | sed 's/^./  miss margins: &/')"
    a=$(echo "$line" | sed 's/.*: \([0-9]*\)\/\([0-9]*\).*/\1/'); b=$(echo "$line" | sed 's/.*\/\([0-9]*\).*/\1/')
    agree=$((agree + a)); total=$((total + b))
  fi
done
[ "$total" -gt 0 ] && python3 -c "print(f'teacher-forced top-1 agreement overall: $agree/$total = {100*$agree/$total:.2f}%')"
