#!/bin/bash
#SBATCH --job-name=spring_diagnosis
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=03:00:00
#SBATCH --array=1-45
#SBATCH --output=logs/diag_%A_%a.out
#SBATCH --error=logs/diag_%A_%a.err

# =============================================================================
# Diagnosis experiment for Same-Sampled SPRING (paper Sec. 4.4, fig:diagnosis)
# =============================================================================
#
# QUESTION
#   Adaptive SPRING sets its momentum from the ratio of ||eps_t||^2 summed over
#   two consecutive windows of p steps each. That ratio is meant to measure how
#   fast the *iteration* contracts. But ||eps_t|| also moves when theta moves the
#   optimum, and when the collocation rows change underneath it. On a fixed
#   linear system both confounds vanish -- which is why Adaptive SPRING works
#   there and (hypothesis) fails on PINNs.
#
# MANIPULATION
#   --batch_frequency=k regenerates the collocation rows every k steps
#   (train_utils.py:89-135). k=1 is fresh rows every step, which is what all 180
#   existing sweep YAMLs use. Nothing else varies across the grid.
#
# PREDICTION (this is what the figure must show)
#   The claim is an INTERACTION, not a main effect:
#     adaptive  -- final L2 improves as k rises. Only arm whose control signal
#                  is contaminated by resampling.
#     spring    -- flat in k. Fixed mu; never reads the signal.
#     ss_spring -- flat in k. Signal already resampling-invariant.
#   One sloped line against two flat ones is the result. If all three slope
#   together, the effect is data diversity, not the control signal, and the
#   diagnosis in Sec. 4.4 is WRONG -- which is exactly why this runs first.
#
#   The grid straddles 2p = 60 deliberately. Both comparison windows share one
#   row set only when k >= 60, so the predicted elbow sits at k ~ 60. An elbow
#   landing where the mechanism says it should is worth more than a monotone
#   trend.
#
# INTERPRETIVE CAVEAT -- state this in the caption
#   Raising k also reduces the number of distinct collocation sets seen over the
#   run, which independently hurts the PDE solution. That contaminates the MAIN
#   effect of k for every arm equally, so read the arm-to-arm DIFFERENCE, never
#   a single curve in isolation. The eval grid is unaffected: the eval loader is
#   hard-wired to frequency 0 (train.py:568), so l2_error is measured on the
#   same fixed points regardless of k.
#
# REQUIRES
#   The --SPRING_lb_window flag, added to parse_SPRING_args in optim/spring.py.
#   Without it lb_window is stuck at its constructor default of 30 and the
#   plain-SPRING control arm is unreachable from the CLI.
#
# COST  45 tasks x ~2 h = ~90 GPU-hours, fully parallel as an array.
# =============================================================================

set -euo pipefail

# DRY_RUN=1 prints the command this task would run and exits, skipping the
# cluster environment entirely. Verify all 45 command lines before submitting:
#   for i in $(seq 1 45); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/run_diagnosis.sh; done
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or set it by hand" >&2
  echo "       for a dry run: SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash $0" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  # --- Environment setup ---
  # Batch jobs do NOT read .bashrc by default, so conda must be initialized here.
  source ~/.bashrc
  conda activate rla_pinns || { echo "ERROR: conda activate failed"; exit 1; }

  cd ~/rla-pinns-use/rla_pinns || { echo "ERROR: cd failed"; exit 1; }
fi

# --- Grid: 3 arms x 5 frequencies x 3 seeds = 45 -----------------------------
ARMS=(adaptive spring ss_spring)
FREQS=(1 15 30 60 120)
DATA_SEEDS=(0 1 2)
MODEL_SEEDS=(1 2 3)

i=$(( SLURM_ARRAY_TASK_ID - 1 ))
seed_idx=$(( i % 3 ))
freq_idx=$(( (i / 3) % 5 ))
arm_idx=$(( i / 15 ))

ARM=${ARMS[$arm_idx]}
K=${FREQS[$freq_idx]}
DATA_SEED=${DATA_SEEDS[$seed_idx]}
MODEL_SEED=${MODEL_SEEDS[$seed_idx]}

# --- Shared problem definition ------------------------------------------------
# Matched to runs/run_array_2.sh so results are comparable with the existing
# Poisson-100d figures. A NUMERIC lr is essential: grid_line_search would re-tune
# per step and adapt to the changed data regime, silently absorbing the effect
# this experiment is trying to measure.
LR=0.0924
DAMPING=0.0301
COMMON="--N_Omega=200 --N_dOmega=100 --N_eval=2000 \
  --dim_Omega=100 --equation=poisson --boundary_condition=u_weinan_norm \
  --model=mlp-tanh-768-768-512-512 --num_seconds=7000 \
  --batch_frequency=${K} --data_seed=${DATA_SEED} --model_seed=${MODEL_SEED} \
  --wandb --wandb_project=spring_diagnosis \
  --wandb_name=${ARM}_k${K}_s${DATA_SEED}"

# --- Arms ---------------------------------------------------------------------
case $ARM in
  # The method under test: adaptive momentum driven by ||eps_t||.
  adaptive)
    ARGS="--optimizer=SPRING \
      --SPRING_lr=${LR} --SPRING_damping=${DAMPING} \
      --SPRING_momentum=0.99 --SPRING_lb_window=30 \
      --SPRING_norm_constraint=1e-3 ${COMMON}"
    ;;

  # Control: fixed mu. lb_window=0 sets _use_adaptive_beta=False, so
  # decay_factor keeps its constructor value (spring.py:152) for the whole run.
  # Insensitive to k by construction -- it never reads the progress signal.
  spring)
    ARGS="--optimizer=SPRING \
      --SPRING_lr=${LR} --SPRING_damping=${DAMPING} \
      --SPRING_momentum=0.99 --SPRING_lb_window=0 \
      --SPRING_norm_constraint=1e-3 ${COMMON}"
    ;;

  # The proposed fix: momentum driven by the probe residual instead.
  # --adaptive_probe is REQUIRED. Without it eta_p stays fixed and the probe is
  # not a Kaczmarz++ iteration at all (the omission in run_array_2.sh:32-35).
  ss_spring)
    ARGS="--optimizer=SameSampledSPRINGUnified \
      --SameSampledSPRINGUnified_lr=${LR} \
      --SameSampledSPRINGUnified_damping=${DAMPING} \
      --SameSampledSPRINGUnified_momentum=0.99 \
      --SameSampledSPRINGUnified_lb_window=30 \
      --SameSampledSPRINGUnified_probe_lr=${LR} \
      --SameSampledSPRINGUnified_probe_damping=${DAMPING} \
      --SameSampledSPRINGUnified_adaptive_eta \
      --SameSampledSPRINGUnified_adaptive_probe ${COMMON}"
    ;;

  *)
    echo "ERROR: unknown arm '${ARM}'" >&2; exit 1
    ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} batch_frequency=${K} \
data_seed=${DATA_SEED} model_seed=${MODEL_SEED} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python train.py ${ARGS}
