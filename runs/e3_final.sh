#!/bin/bash
#SBATCH --job-name=e3_final
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=02:00:00
#SBATCH --array=1-18
#SBATCH --output=logs/e3_%A_%a.out
#SBATCH --error=logs/e3_%A_%a.err
# =============================================================================
# E3 -- Final battery. THIS IS THE PAPER'S PINN RESULT (Fig A + Tab A).
# See PINN_EXPERIMENTS.md section 4 / E3.
# =============================================================================
#
# 3 arms x 2 PDEs x 3 model_seeds = 18 runs at matched wall-clock. Each arm uses
# ITS OWN E2 winner, read from runs/best_hparams.csv -- this script refuses to
# run if that file is missing, so E3 can never quietly execute on hand-typed
# numbers.
#
# THREE SEEDS IS A FLOOR, NOT A CHOICE
#   tab:exp-pinns promises a standard deviation and ZERO seeds exist anywhere in
#   the repo. There is no version of that table assemblable from existing data.
#
# EQUAL WALL-CLOCK, NOT EQUAL STEPS
#   SS-SPRING's probe costs extra per step. An equal-step comparison would hand
#   it a budget advantage it does not have in practice. Report measured s/step
#   alongside the errors so the per-step overhead is visible, not hidden.
#
# BEFORE SUBMITTING
#   Size --time from E1's MEASURED s/step, not from --num_seconds. Archived 100d
#   runs at N_total=625 measured 16.4-16.7 s/step and two of them terminated at
#   ~46% of their declared budget for reasons never explained. --num_seconds=5000
#   under --time=02:00:00 leaves ~44% headroom; widen it if E1 says you need to.
#
# KILL CRITERION
#   If SS-SPRING does not beat tuned fixed-mu SPRING (a1) by more than E1(c)'s
#   metric noise floor on BOTH PDEs, C2 is not established. Do not report a win
#   on one PDE as a general one -- name the PDE and say why the other differs.
#
# DRY RUN (works without best_hparams.csv only if you pass ALLOW_MISSING_CSV=1):
#   for i in $(seq 1 18); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e3_final.sh; done
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
HPARAMS="${HPARAMS:-runs/best_hparams.csv}"
ALLOW_MISSING_CSV="${ALLOW_MISSING_CSV:-0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or for a dry run:" >&2
  echo "       SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 ALLOW_MISSING_CSV=1 bash $0" >&2
  exit 1
fi

# --- Grid: 3 seeds x 2 PDEs x 3 arms = 18 ------------------------------------
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
seed_idx=$(( i % 3 ))
pde_idx=$(( (i / 3) % 2 ))
arm_idx=$(( i / 6 ))

MODEL_SEEDS=(1 2 3)
ARMS=(a1_spring a2_adaptive a3_ss)
MODEL_SEED=${MODEL_SEEDS[$seed_idx]}
ARM=${ARMS[$arm_idx]}
if [[ $pde_idx -eq 0 ]]; then PDE=p100; else PDE=lfp9; fi

# --- Hyperparameters: read this arm's E2 winner ------------------------------
# Columns: arm,pde,lr,damping,momentum,l2_error,draw,n_usable,log
if [[ -f "$HPARAMS" ]]; then
  ROW=$(awk -F, -v a="$ARM" -v p="$PDE" \
    'NR>1 && $1==a && $2==p {print $3","$4","$5; found=1} END{if(!found) exit 3}' \
    "$HPARAMS") || {
      echo "ERROR: no row for arm=$ARM pde=$PDE in $HPARAMS." >&2
      echo "       E3 needs all six (arm, pde) cells. Re-run runs/harvest_e2.py" >&2
      echo "       and check for cells where every draw diverged." >&2
      exit 1
    }
  LR=$(echo "$ROW" | cut -d, -f1)
  DAMPING=$(echo "$ROW" | cut -d, -f2)
  MOMENTUM=$(echo "$ROW" | cut -d, -f3)
elif [[ "$ALLOW_MISSING_CSV" == "1" ]]; then
  # Dry-run placeholders ONLY. Deliberately absurd so that a real run started
  # without the csv is obvious in the logs rather than silently plausible.
  LR=PLACEHOLDER_LR
  DAMPING=PLACEHOLDER_DAMPING
  MOMENTUM=PLACEHOLDER_MOMENTUM
else
  echo "ERROR: $HPARAMS not found." >&2
  echo "       E3 must run on E2's tuned winners. Generate it with:" >&2
  echo "         python runs/harvest_e2.py --logs logs --out runs/best_hparams.csv" >&2
  echo "       (to inspect command lines without it: ALLOW_MISSING_CSV=1 DRY_RUN=1)" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$LR" == PLACEHOLDER_* ]]; then
    echo "ERROR: refusing to train on placeholder hyperparameters." >&2
    exit 1
  fi
  mkdir -p logs
  set +eu
  source ~/.bashrc
  conda activate rla_pinns
  conda_status=$?
  set -eu
  if [[ $conda_status -ne 0 ]]; then
    echo "ERROR: conda activate rla_pinns failed (status $conda_status)" >&2
    exit 1
  fi
  cd ~/rla-pinns-use/rla_pinns || { echo "ERROR: cd failed" >&2; exit 1; }
fi

P100="--equation=poisson --boundary_condition=u_weinan_norm --dim_Omega=100 \
--model=mlp-tanh-768-768-512-512 --N_Omega=200 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"

LFP9="--equation=log-fokker-planck-isotropic --boundary_condition=gaussian \
--dim_Omega=9 --model=mlp-tanh-256-256-128-128 --N_Omega=300 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"

if [[ "$PDE" == "p100" ]]; then CFG="$P100"; else CFG="$LFP9"; fi

TAG="${ARM}_${PDE}_m${MODEL_SEED}"
COMMON="${CFG} --num_seconds=5000 --model_seed=${MODEL_SEED} --max_logs=150 \
--wandb --wandb_entity=rla-pinns --wandb_project=pinn_e3_final --wandb_name=${TAG}"

case $ARM in
  a1_spring)
    ARGS="--optimizer=SPRING \
--SPRING_lr=${LR} --SPRING_damping=${DAMPING} --SPRING_momentum=${MOMENTUM} \
--SPRING_lb_window=0 --SPRING_norm_constraint=1e-3 --SPRING_beta_max=0.99 \
${COMMON}"
    ;;
  a2_adaptive)
    ARGS="--optimizer=SPRING \
--SPRING_lr=${LR} --SPRING_damping=${DAMPING} --SPRING_momentum=${MOMENTUM} \
--SPRING_lb_window=30 --SPRING_norm_constraint=1e-3 --SPRING_beta_max=0.99 \
${COMMON}"
    ;;
  a3_ss)
    ARGS="--optimizer=SameSampledSPRINGUnified \
--SameSampledSPRINGUnified_lr=${LR} \
--SameSampledSPRINGUnified_damping=${DAMPING} \
--SameSampledSPRINGUnified_momentum=${MOMENTUM} \
--SameSampledSPRINGUnified_lb_window=30 \
--SameSampledSPRINGUnified_probe_lr=${LR} \
--SameSampledSPRINGUnified_probe_damping=${DAMPING} \
--SameSampledSPRINGUnified_norm_constraint=1e-3 \
--SameSampledSPRINGUnified_beta_max=0.99 \
--SameSampledSPRINGUnified_probe_seed=0 \
--SameSampledSPRINGUnified_adaptive_eta \
--SameSampledSPRINGUnified_adaptive_probe \
${COMMON}"
    ;;
  *) echo "ERROR: unknown arm '${ARM}'" >&2; exit 1 ;;
esac

echo "=== E3 task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} pde=${PDE} \
model_seed=${MODEL_SEED} lr=${LR} damping=${DAMPING} momentum=${MOMENTUM} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python -u train.py ${ARGS}
