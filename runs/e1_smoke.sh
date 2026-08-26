#!/bin/bash
#SBATCH --job-name=e1_smoke
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=00:20:00
#SBATCH --array=1-7
#SBATCH --output=logs/e1_%A_%a.out
#SBATCH --error=logs/e1_%A_%a.err
# =============================================================================
# E1 -- Smoke + calibration.  See PINN_EXPERIMENTS.md section 4 / E1.
# =============================================================================
#
# PURPOSE
#   Cheap insurance before committing 52 GPU-h to E2. Three things:
#     (a) tasks 1-6: do all three arms run on both PDEs, and do `l2_error` and
#         `decay_factor` (= beta) actually appear in wandb?
#     (b) tasks 1-6 also give the measured s/step per (arm, PDE), which is what
#         sizes --time for E2 and E3. Do not size --time from --num_seconds.
#     (c) task 7: produce ONE checkpoint from which the offline noise floor of
#         l2_error is measured (runs/e1_noise_floor.py). Written to ../ckpt_noise
#         so it lands at the repo root next to logs/, not inside the package dir
#         (python runs with cwd=~/rla-pinns-use/rla_pinns). That number is the
#         minimum reportable difference in Tab A -- if E3's SS-SPRING vs tuned
#         SPRING gap lands inside it, C2 is not established.
#
# PREREQUISITE
#   Commit the working tree first. The private-RNG fix for x_star is what makes
#   SS-SPRING and SPRING at the same --model_seed see the SAME collocation data.
#   Without it every comparison in E3 is confounded. See section 3.
#
# DRY RUN -- verify all 7 command lines without touching the cluster:
#   for i in $(seq 1 7); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e1_smoke.sh; done
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or for a dry run:" >&2
  echo "       SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash $0" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  # SLURM will NOT create the --output directory; without this an array can die
  # instantly with no log explaining why. Harmless when it already exists.
  mkdir -p logs

  # `set -u` must be OFF across this block: a stock ~/.bashrc opens with
  # `[ -z "$PS1" ] && return` and PS1 is unbound non-interactively, so the
  # source aborts the job at 00:00:00. conda's shell functions trip the same way.
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

# --- Problem configs ---------------------------------------------------------
# N_eval is pinned EXPLICITLY everywhere. The default is 10*N_Omega, so it
# silently tracks the batch size; 179 of 180 existing sweep YAMLs have that bug.
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

P100="--equation=poisson --boundary_condition=u_weinan_norm --dim_Omega=100 \
--model=mlp-tanh-768-768-512-512 --N_Omega=200 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"

# N_Omega=300, NOT the archived 3000. A 4000-row batch has ~4x less sampling
# noise, i.e. it is precisely where the mechanism should be WEAKEST. See sec 2.
LFP9="--equation=log-fokker-planck-isotropic --boundary_condition=gaussian \
--dim_Omega=9 --model=mlp-tanh-256-256-128-128 --N_Omega=300 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"

# Inherited anchor: every SPRING-family run to date used this pair, itself taken
# from a DIFFERENT optimizer's sweep winner. E2 re-tunes; here we just smoke it.
LR=0.0924
DAMPING=0.0301
MOMENTUM=0.99

i=$(( SLURM_ARRAY_TASK_ID - 1 ))

if [[ $SLURM_ARRAY_TASK_ID -le 6 ]]; then
  ARMS=(a1_spring a2_adaptive a3_ss)
  ARM=${ARMS[$(( i % 3 ))]}
  if [[ $(( i / 3 )) -eq 0 ]]; then PDE=p100; CFG="$P100"; else PDE=lfp9; CFG="$LFP9"; fi
  BUDGET="--num_seconds=600"
  EXTRA=""
  TAG="smoke_${ARM}_${PDE}"
else
  # Task 7: checkpoint producer for the offline metric-noise floor.
  # Step-budgeted with an EXPLICIT --checkpoint_steps on purpose: in that mode
  # `should_log` is a pure set-membership test, so the stateful-closure bug that
  # corrupts logging cadence in wall-clock mode cannot fire.
  ARM=a1_spring; PDE=p100; CFG="$P100"
  BUDGET="--num_steps=200"
  EXTRA="--save_checkpoints --checkpoint_steps 199 --checkpoint_dir=../ckpt_noise"
  TAG="ckpt_${ARM}_${PDE}"
fi

COMMON="${CFG} ${BUDGET} ${EXTRA} --model_seed=1 --max_logs=150 \
--wandb ${ENTITY_ARG} --wandb_project=pinn_e1_smoke --wandb_name=${TAG}"

# --- Arms --------------------------------------------------------------------
# beta_max is pinned on BOTH classes: defaults are asymmetric (SPRING 1.0
# uncapped, Unified 0.99), so leaving them default confounds the arms.
case $ARM in
  # A1: fixed mu. lb_window=0 -> _use_adaptive_beta=False, so decay_factor keeps
  # its constructor value all run. This is the baseline that matters for C2.
  a1_spring)
    ARGS="--optimizer=SPRING \
--SPRING_lr=${LR} --SPRING_damping=${DAMPING} --SPRING_momentum=${MOMENTUM} \
--SPRING_lb_window=0 --SPRING_norm_constraint=1e-3 --SPRING_beta_max=0.99 \
${COMMON}"
    ;;
  # A2: the method under test -- momentum driven by the sampled PDE residual.
  a2_adaptive)
    ARGS="--optimizer=SPRING \
--SPRING_lr=${LR} --SPRING_damping=${DAMPING} --SPRING_momentum=${MOMENTUM} \
--SPRING_lb_window=30 --SPRING_norm_constraint=1e-3 --SPRING_beta_max=0.99 \
${COMMON}"
    ;;
  # A3: ours. adaptive_eta AND adaptive_probe are both required -- without
  # adaptive_probe, eta_pr stays at probe_lr and the probe is not a Kaczmarz++
  # iteration at all. Never use --optimizer=SameSampledSPRING (no trust region).
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

echo "=== E1 task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} pde=${PDE} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

# -u is deliberate: r_hat and rho reach stdout only via print(), so a buffered
# stream would lose them if the job is killed.
python -u train.py ${ARGS}
