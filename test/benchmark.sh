#!/usr/bin/env bash
set -euo pipefail

# Where to store all runs
METRICS_FILE="metrics.txt"
# Remove old results so header is recreated
rm -f "$METRICS_FILE"

# Define your (n,l) combinations
# Define your (n,l) combinations here:
COMBOS=(
  "1000   200"
  "1000   400"
  "1000   600"
  "1000   800"
  "5000   1000"
  "5000   2000"
  "5000   3000"
  "5000   4000"
  "10000  2000"
  "10000  4000"
  "10000  6000"
  "10000  8000"
)

# Number of iterations inside each Python benchmark
ITERATIONS=20

for combo in "${COMBOS[@]}"; do
  read -r n l <<<"$combo"
  echo "=== Benchmarking: n=$n, l=$l ==="
  python test_nystrom.py \
    --n "$n" \
    --l "$l" \
    --iterations "$ITERATIONS" \
    --output "$METRICS_FILE"
  echo
done

echo "✅ All benchmarks complete. Results in $METRICS_FILE"
