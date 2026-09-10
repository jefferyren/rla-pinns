#!/bin/bash
# =============================================================================
# E5 -- submission wrapper. Submits the per-system arrays with the right
# --time, because a shared wall-clock across four systems would either truncate
# lfp9/p100 or reserve three hours to run heat4 for fifty minutes.
#
#   bash runs/e5_submit.sh smoke            # 16 tasks,  ~3 GPU-h   (run first)
#   bash runs/e5_submit.sh tune             # 4 arrays, 192 tasks, ~73 GPU-h
#   bash runs/e5_submit.sh final            # 4 arrays,  48 tasks, ~87 GPU-h
#   bash runs/e5_submit.sh tune  heat4 lfp9 # only those systems
#   DRY=1 bash runs/e5_submit.sh final      # print the sbatch lines, submit nothing
#
# --time values carry ~20% headroom over --num_seconds. Re-derive them from
# E5(a)'s measured s/step before submitting `final`: archived 100d runs in this
# repo measured 16.4-16.7 s/step and two terminated at ~46% of their declared
# budget for reasons never explained.
# =============================================================================

set -euo pipefail

STAGE="${1:-}"
shift || true
DRY="${DRY:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- Per-system wall-clock ---------------------------------------------------
#                        tune       final      (num_seconds: tune / final)
#   p5                00:35:00   02:20:00      1400 /  7000
#   p100              00:50:00   03:20:00      2000 / 10000
#   heat4             00:25:00   01:05:00       900 /  3000
#   lfp9              00:30:00   02:05:00      1200 /  6000
tune_time()  { case "$1" in p5) echo 00:35:00;; p100) echo 00:50:00;; heat4) echo 00:25:00;; lfp9) echo 00:30:00;; esac; }
final_time() { case "$1" in p5) echo 02:20:00;; p100) echo 03:20:00;; heat4) echo 01:05:00;; lfp9) echo 02:05:00;; esac; }

ALL_SYSTEMS=(p5 p100 heat4 lfp9)
if (( $# > 0 )); then
  SYSTEMS=("$@")
else
  SYSTEMS=("${ALL_SYSTEMS[@]}")
fi
for s in "${SYSTEMS[@]}"; do
  case "$s" in p5|p100|heat4|lfp9) ;; *) echo "ERROR: unknown system '$s'" >&2; exit 1;; esac
done

run() {
  echo "+ $*"
  if [[ "$DRY" != "1" ]]; then "$@"; fi
}

case "$STAGE" in
  smoke)
    # One array covers all four systems: every task is capped at 600s anyway.
    run sbatch --time=00:25:00 "${HERE}/e5_smoke.sh"
    ;;
  tune)
    for s in "${SYSTEMS[@]}"; do
      run sbatch --export="ALL,SYS=${s}" --time="$(tune_time "$s")" \
        --job-name="e5tune_${s}" "${HERE}/e5_tune.sh"
    done
    ;;
  final)
    CSV="${HERE}/e5_best_hparams.csv"
    [[ -f "$CSV" ]] || CSV="${E5_REPO:-$HOME/rla-pinns-use}/runs/e5_best_hparams.csv"
    if [[ ! -f "$CSV" && "$DRY" != "1" ]]; then
      echo "ERROR: runs/e5_best_hparams.csv is missing." >&2
      echo "       Run E5(b) first, then:" >&2
      echo "         python runs/harvest_e5.py --logs logs --out runs/e5_best_hparams.csv" >&2
      exit 1
    fi
    for s in "${SYSTEMS[@]}"; do
      run sbatch --export="ALL,SYS=${s}" --time="$(final_time "$s")" \
        --job-name="e5final_${s}" "${HERE}/e5_final.sh"
    done
    ;;
  *)
    echo "usage: bash runs/e5_submit.sh {smoke|tune|final} [system ...]" >&2
    echo "       systems: p5 p100 heat4 lfp9" >&2
    exit 1
    ;;
esac
