#!/bin/bash
#SBATCH --job-name=e5_final
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=03:20:00
#SBATCH --array=1-12
#SBATCH --output=logs/e5final_%A_%a.out
#SBATCH --error=logs/e5final_%A_%a.err
# =============================================================================
# E5(c) -- Final battery. THIS IS THE FOUR-PANEL FIGURE.
# See PINN_EXPERIMENTS.md section 4 / E5.
#
# ONE SYSTEM PER SUBMISSION -- each has its own time budget:
#
#   sbatch --export=ALL,SYS=heat4 --time=01:15:00 runs/e5_final.sh
#
# runs/e5_submit.sh final  does all four with the right --time. Use that.
# =============================================================================
#
# 4 arms x 3 model_seeds = 12 runs per system, at the paper's own wall-clock
# budget for that system (p5 7000s, p100 10000s, heat4 3000s, lfp9 6000s).
# Across all four systems: 48 runs, ~87 GPU-h.
#
# THIS IS A REPLICATION PLUS TWO ARMS, NOT A REPLICATION
#   Panels, architectures, batch sizes, boundary conditions, budgets, and the
#   engdw/spring hyperparameters are arXiv:2505.12149's Figure 3 exactly. What
#   changes: its "ENGD-W (Line Search)" and "KFAC" arms are dropped, and
#   primesr and ssspring take their place. Anyone reading the figure will
#   compare it to Figure 3, so the caption must say which two arms were
#   swapped out and that the remaining two are at the published values.
#
# THREE SEEDS, WHERE THE PAPER REPORTS ONE
#   Figure 3 plots the single best run of each sweep. That is fine for "our
#   method reaches a lower error"; it is not enough to separate four arms whose
#   curves may sit within seed noise of each other. Seeds 1-3 are held out from
#   E5(b)'s selection, which used seed 101. Report median with a min-max band,
#   and put the seed count in the caption.
#
# EQUAL WALL-CLOCK, NOT EQUAL STEPS
#   ssspring's probe and primesr's per-step eigendecomposition both cost extra
#   per step. Equal-step would hand them a budget advantage they do not have in
#   practice. Report measured s/step alongside the errors so the overhead is
#   visible rather than hidden.
#
# KNOWN ASYMMETRY, DO NOT PAPER OVER IT
#   engdw and spring run with NO trust region -- optim/rngd.py has no
#   norm_constraint and train.py's check_all_args_parsed() rejects the flag
#   outright. primesr and ssspring both have one. This is inherent to running
#   the paper's arms as published; name it in the write-up.
#
# BEFORE SUBMITTING
#   1. E5(a) smoke must be clean, especially the lfp9 memory check and the
#      heat4 hyperparameter check -- both are described in e5_smoke.sh.
#   2. runs/e5_best_hparams.csv must exist. This script refuses to run without
#      it, so E5(c) can never quietly execute on hand-typed numbers.
#   3. Size --time from E5(a)'s MEASURED s/step, not from --num_seconds. The
#      defaults in e5_submit.sh leave ~20% headroom over the budget.
#
# DRY RUN (needs ALLOW_MISSING_CSV=1 before tuning has landed):
#   for i in $(seq 1 12); do SLURM_ARRAY_TASK_ID=$i SYS=p5 DRY_RUN=1 \
#     ALLOW_MISSING_CSV=1 bash runs/e5_final.sh; done
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
E5_REPO="${E5_REPO:-$HOME/rla-pinns-use}"
# Resolved against the checkout, not the submission directory: sbatch sets the
# job's cwd to wherever it was launched from, and a relative "runs/..." would
# then silently miss and trip the not-found guard on an otherwise fine run.
HPARAMS="${HPARAMS:-}"
if [[ -z "$HPARAMS" ]]; then
  for cand in "${E5_REPO}/runs/e5_best_hparams.csv" \
              "$(cd "$(dirname "$0")" && pwd)/e5_best_hparams.csv" \
              "runs/e5_best_hparams.csv"; do
    if [[ -f "$cand" ]]; then HPARAMS="$cand"; break; fi
  done
  HPARAMS="${HPARAMS:-${E5_REPO}/runs/e5_best_hparams.csv}"
fi
ALLOW_MISSING_CSV="${ALLOW_MISSING_CSV:-0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or for a dry run:" >&2
  echo "       SLURM_ARRAY_TASK_ID=1 SYS=p5 DRY_RUN=1 ALLOW_MISSING_CSV=1 bash $0" >&2
  exit 1
fi
if [[ -z "${SYS:-}" ]]; then
  echo "ERROR: SYS unset. This script runs ONE system per submission:" >&2
  echo "       sbatch --export=ALL,SYS=p5 --time=02:20:00 runs/e5_final.sh" >&2
  echo "       (or just use runs/e5_submit.sh final)" >&2
  exit 1
fi

SYSTEMS_FILE="${E5_REPO}/runs/e5_systems.sh"
[[ -f "$SYSTEMS_FILE" ]] || SYSTEMS_FILE="$(cd "$(dirname "$0")" && pwd)/e5_systems.sh"
[[ -f "$SYSTEMS_FILE" ]] || { echo "ERROR: cannot find e5_systems.sh" >&2; exit 1; }
# shellcheck source=/dev/null
source "$SYSTEMS_FILE"

e5_sys_config "$SYS"
e5_paper_hparams "$SYS"

# --- Grid: 4 arms x 3 seeds = 12 ---------------------------------------------
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
seed_idx=$(( i % 3 ))
arm_idx=$(( i / 3 ))
if (( arm_idx > 3 )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside 1-12" >&2
  exit 1
fi
MODEL_SEEDS=(1 2 3)
MODEL_SEED=${MODEL_SEEDS[$seed_idx]}
ARM=${E5_ARMS[$arm_idx]}

# --- Hyperparameters ---------------------------------------------------------
# engdw / spring: the paper's published fixed-lr values, never the csv.
# primesr / ssspring: this arm's E5(b) winner, read from the csv.
NC=1e-3
case $ARM in
  engdw)
    LR=$PAPER_ENGDW_LR; DAMPING=$PAPER_ENGDW_DAMPING; MOMENTUM=0.0
    SOURCE=paper
    ;;
  spring)
    LR=$PAPER_SPRING_LR; DAMPING=$PAPER_SPRING_DAMPING; MOMENTUM=$PAPER_SPRING_MOMENTUM
    SOURCE=paper
    ;;
  primesr|ssspring)
    SOURCE=tuned
    # Columns: arm,sys,lr,damping,momentum,norm_constraint,l2_error,draw,n_usable,log
    if [[ -f "$HPARAMS" ]]; then
      ROW=$(awk -F, -v a="$ARM" -v s="$SYS" \
        'NR>1 && $1==a && $2==s {print $3","$4","$5","$6; found=1} END{if(!found) exit 3}' \
        "$HPARAMS") || {
          echo "ERROR: no row for arm=${ARM} sys=${SYS} in ${HPARAMS}." >&2
          echo "       E5(c) needs all eight (arm, system) cells. Re-run" >&2
          echo "       runs/harvest_e5.py and check for cells where every" >&2
          echo "       draw diverged." >&2
          exit 1
        }
      LR=$(echo "$ROW" | cut -d, -f1)
      DAMPING=$(echo "$ROW" | cut -d, -f2)
      MOMENTUM=$(echo "$ROW" | cut -d, -f3)
      NC=$(echo "$ROW" | cut -d, -f4)
    elif [[ "$ALLOW_MISSING_CSV" == "1" ]]; then
      # Dry-run placeholders ONLY. Deliberately absurd so a real run started
      # without the csv is obvious in the log rather than silently plausible.
      LR=PLACEHOLDER_LR
      DAMPING=PLACEHOLDER_DAMPING
      MOMENTUM=PLACEHOLDER_MOMENTUM
      NC=PLACEHOLDER_NC
    else
      echo "ERROR: ${HPARAMS} not found." >&2
      echo "       E5(c) must run on E5(b)'s tuned winners. Generate it with:" >&2
      echo "         python runs/harvest_e5.py --logs logs --out runs/e5_best_hparams.csv" >&2
      echo "       (to inspect command lines without it: ALLOW_MISSING_CSV=1 DRY_RUN=1)" >&2
      exit 1
    fi
    ;;
esac

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$LR" == PLACEHOLDER_* ]]; then
    echo "ERROR: refusing to train on placeholder hyperparameters." >&2
    exit 1
  fi
  e5_activate
fi
e5_entity_arg
e5_arm_args "$ARM" "$LR" "$DAMPING" "$MOMENTUM" "$NC"

TAG="${ARM}_${SYS}_m${MODEL_SEED}"
ARGS="${ARM_ARGS} ${CFG} --num_seconds=${BUDGET} --model_seed=${MODEL_SEED} --max_logs=150 \
--wandb ${ENTITY_ARG} --wandb_project=pinn_e5_final --wandb_name=${TAG}"

echo "=== E5-FINAL task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} sys=${SYS} \
model_seed=${MODEL_SEED} source=${SOURCE} lr=${LR} damping=${DAMPING} \
momentum=${MOMENTUM} nc=${NC} budget=${BUDGET} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python -u train.py ${ARGS}
