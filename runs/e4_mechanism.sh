#!/bin/bash
#SBATCH --job-name=e4_mech
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=01:30:00
#SBATCH --array=1-18
#SBATCH --output=logs/e4_%A_%a.out
#SBATCH --error=logs/e4_%A_%a.err
# =============================================================================
# E4 -- Mechanism figure (Fig B).  OPTIONAL. See PINN_EXPERIMENTS.md sec 4 / E4.
# =============================================================================
#
# DECIDE BEFORE RUNNING. Without E4 the paper says: Adaptive SPRING fails on
# PINNs, same-sampling fixes it -- an empirical claim, fully supported by E2+E3,
# with the mechanism left as a hypothesis in prose. With E4 the paper can say
# WHY, and SS-SPRING stops looking like an arbitrary trick that happened to work.
# 27 GPU-h.
#
# THE KNOB
#   --batch_frequency=k regenerates the collocation rows when step % k == 0.
#   k=0 means DRAW ONCE AT STEP 0 AND REUSE FOREVER -- the true fixed-batch
#   control, and it has NEVER been run in this repo: 177 of 180 sweep YAMLs pin
#   batch_frequency=1 and all 3 exceptions are KFAC. k=60 = 2p is the first value
#   at which both of the controller's comparison windows share one row set.
#
# THE PREDICTION (this is what the figure must show)
#   An INTERACTION, not a main effect:
#     a2_adaptive -- final l2_error improves as k rises. The only arm whose
#                    control signal is contaminated by resampling.
#     a1_spring   -- flat in k. Fixed mu; never reads the signal.
#     a3_ss       -- flat in k. Signal already resampling-invariant.
#   One sloped line against two flat ones. If a2 at k=0 matches a1, the failure
#   is CAUSED BY the rows changing under the measurement -- exactly what
#   same-sampling removes.
#
# INTERPRETIVE CAVEAT -- put this in the caption
#   Raising k also reduces the number of distinct collocation sets seen over the
#   run, which independently hurts the PDE solution. That contaminates the MAIN
#   effect of k for every arm equally, so read the arm-to-arm DIFFERENCE, never a
#   single curve alone. The eval grid is unaffected: the eval loader is hard-wired
#   to frequency 0, so l2_error is measured on the same fixed points at every k.
#
# PLOT BETA, NOT JUST LOSS
#   decay_factor (= beta) is already logged to wandb for all three arms
#   (train.py:826). Loss curves show THAT a2 recovers; beta shows the controller
#   doing the thing the paper claims it does. r_hat and rho reach stdout only,
#   parseable with the RE_BETA regex in runs/harvest_diagnosis.py.
#
# KILL CRITERION
#   If a1 and a3 also improve substantially with k, large k simply helps any
#   momentum method and the figure shows nothing specific about the controller:
#   drop Fig B and keep the mechanism out of the abstract. If a2 does NOT recover
#   at k=0, the stated mechanism is wrong -- report that; C1/C2 stand on E2+E3.
#
# NOTE  This supersedes runs/run_diagnosis.sh, which is dead as written: it passes
#       --data_seed in {0,1,2} and train.py now raises on any non-zero value, so
#       30 of its 45 tasks exit before the first step. Its header comment remains
#       the best statement of the mechanism argument and is worth keeping.
#
# DRY RUN:
#   for i in $(seq 1 18); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 ALLOW_MISSING_CSV=1 bash runs/e4_mechanism.sh; done
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

# --- Grid: 2 seeds x 3 frequencies x 3 arms = 18 -----------------------------
# k=0 and k=60 are the decisive columns. k=30 (= p) is deliberately NOT included:
# it is the ambiguous midpoint, and the budget buys k=0 instead.
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
seed_idx=$(( i % 2 ))
k_idx=$(( (i / 2) % 3 ))
arm_idx=$(( i / 6 ))

MODEL_SEEDS=(1 2)
FREQS=(0 1 60)
ARMS=(a1_spring a2_adaptive a3_ss)
MODEL_SEED=${MODEL_SEEDS[$seed_idx]}
K=${FREQS[$k_idx]}
ARM=${ARMS[$arm_idx]}

# --- Hyperparameters: this arm's E2 winner on P100 ----------------------------
if [[ -f "$HPARAMS" ]]; then
  ROW=$(awk -F, -v a="$ARM" \
    'NR>1 && $1==a && $2=="p100" {print $3","$4","$5; found=1} END{if(!found) exit 3}' \
    "$HPARAMS") || {
      echo "ERROR: no p100 row for arm=$ARM in $HPARAMS." >&2; exit 1; }
  LR=$(echo "$ROW" | cut -d, -f1)
  DAMPING=$(echo "$ROW" | cut -d, -f2)
  MOMENTUM=$(echo "$ROW" | cut -d, -f3)
elif [[ "$ALLOW_MISSING_CSV" == "1" ]]; then
  LR=PLACEHOLDER_LR; DAMPING=PLACEHOLDER_DAMPING; MOMENTUM=PLACEHOLDER_MOMENTUM
else
  echo "ERROR: $HPARAMS not found. Run runs/harvest_e2.py first." >&2
  echo "       (to inspect command lines: ALLOW_MISSING_CSV=1 DRY_RUN=1)" >&2
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

# --- wandb entity -------------------------------------------------------------
# Omitted by default: train.py passes entity=None to wandb.init, which resolves
# to YOUR personal entity. Do NOT hard-code "rla-pinns" -- that is the original
# authors' team and a non-member gets 403 Forbidden at wandb.init, which kills
# the job ~11s in, after the model is built but before the first step.
# To log to a team you belong to:  PINN_WANDB_ENTITY=my-team sbatch runs/<script>
ENTITY_ARG=""
if [[ -n "${PINN_WANDB_ENTITY:-}" ]]; then
  ENTITY_ARG="--wandb_entity=${PINN_WANDB_ENTITY}"
fi

# P100, with batch_frequency as the varied axis.
CFG="--equation=poisson --boundary_condition=u_weinan_norm --dim_Omega=100 \
--model=mlp-tanh-768-768-512-512 --N_Omega=200 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=${K}"

TAG="${ARM}_k${K}_m${MODEL_SEED}"
COMMON="${CFG} --num_seconds=3000 --model_seed=${MODEL_SEED} --max_logs=150 \
--wandb ${ENTITY_ARG} --wandb_project=pinn_e4_mechanism \
--wandb_name=${TAG}"

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

echo "=== E4 task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} batch_frequency=${K} \
model_seed=${MODEL_SEED} lr=${LR} damping=${DAMPING} momentum=${MOMENTUM} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python -u train.py ${ARGS}
