#!/bin/bash
#SBATCH --job-name=e5_smoke
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=00:25:00
#SBATCH --array=1-16
#SBATCH --output=logs/e5smoke_%A_%a.out
#SBATCH --error=logs/e5smoke_%A_%a.err
# =============================================================================
# E5(a) -- Smoke. 4 systems x 4 arms x 600s. RUN THIS FIRST.
# See PINN_EXPERIMENTS.md section 4 / E5.
# =============================================================================
#
# WHAT IT IS FOR -- three questions, all of which otherwise cost a whole sweep:
#
# 1) DOES PRIME-SR EVEN RUN? optim/prime_sr.py has no test in test/optim and no
#    run anywhere in this repo's history. Every other arm here has been
#    exercised. If PRIME-SR is broken, learn it in 25 minutes, not after 73
#    GPU-h of tuning.
#
# 2) DOES lfp9 / heat4 FIT IN 11 GiB? The paper ran on RTX 6000 (24 GiB); the
#    GTX2080TI has 11. lfp9 is the binding case: N = N_Omega + N_dOmega = 4000
#    rows against D = 118 145 parameters, so the Jacobian alone is
#    4000 x 118145 x 8 B = 3.8 GiB in float64, before any workspace. heat4 is
#    3500 x 116865 x 8 B = 3.3 GiB. Both should fit; neither has been run here.
#    A CUDA OOM in this job means E5 needs a smaller batch on that system --
#    which is a config change to e5_systems.sh and a deviation to write up, NOT
#    something to discover inside e5_final.
#
# 3) DO THE PAPER'S HYPERPARAMETERS BEHAVE ON OUR HARDWARE? engdw and spring
#    run here at the paper's published values. Their 600s l2_error should be on
#    the trajectory of the matching Figure 3 panel. If a curve is flat or NaN,
#    the config is wrong and no amount of tuning downstream will fix it.
#
# THE heat4 CHECK IS THE ONE THAT MATTERS MOST. The paper's fixed-lr heat
# hyperparameters (A.5.1) are printed above a figure titled "5d Heat
# (D = 117121)" -- a 6-input network, NOT the 4+1d one (D = 116865) that
# Figure 3 plots. Either the figure is mislabelled or the hyperparameters
# belong to a different problem, and the paper does not say which. If heat4's
# engdw/spring curves here do not track Figure 3, treat the heat panel's
# baselines as untuned and re-tune them with exp15_heat4d_fixed's ranges
# (damping LU([1e-7, 1e-3]), lr LU([1e-3, 1e-1])) before running E5(c).
#
# primesr and ssspring run at the anchor -- the paper's SPRING values for that
# system, with norm_constraint 1e-3. This is draw 1 of e5_tune.sh, so a smoke
# result here is a free first tuning draw.
#
# DRY RUN -- print all 16 command lines without touching the cluster:
#   for i in $(seq 1 16); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e5_smoke.sh; done
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
E5_REPO="${E5_REPO:-$HOME/rla-pinns-use}"
SECS="${E5_SMOKE_SECONDS:-600}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or for a dry run:" >&2
  echo "       SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash $0" >&2
  exit 1
fi

SYSTEMS_FILE="${E5_REPO}/runs/e5_systems.sh"
[[ -f "$SYSTEMS_FILE" ]] || SYSTEMS_FILE="$(cd "$(dirname "$0")" && pwd)/e5_systems.sh"
[[ -f "$SYSTEMS_FILE" ]] || { echo "ERROR: cannot find e5_systems.sh" >&2; exit 1; }
# shellcheck source=/dev/null
source "$SYSTEMS_FILE"

# --- Grid: 4 arms x 4 systems = 16 -------------------------------------------
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
arm_idx=$(( i % 4 ))
sys_idx=$(( i / 4 ))
if (( sys_idx > 3 )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside 1-16" >&2
  exit 1
fi
ARM=${E5_ARMS[$arm_idx]}
SYS=${E5_SYSTEMS[$sys_idx]}

e5_sys_config "$SYS"
e5_paper_hparams "$SYS"

case $ARM in
  engdw)    LR=$PAPER_ENGDW_LR;  DAMPING=$PAPER_ENGDW_DAMPING;  MOMENTUM=0.0 ;;
  spring)   LR=$PAPER_SPRING_LR; DAMPING=$PAPER_SPRING_DAMPING; MOMENTUM=$PAPER_SPRING_MOMENTUM ;;
  *)        LR=$PAPER_SPRING_LR; DAMPING=$PAPER_SPRING_DAMPING; MOMENTUM=$PAPER_SPRING_MOMENTUM ;;
esac
NC=1e-3

if [[ "$DRY_RUN" != "1" ]]; then
  e5_activate
fi
e5_entity_arg
e5_arm_args "$ARM" "$LR" "$DAMPING" "$MOMENTUM" "$NC"

TAG="${ARM}_${SYS}_smoke"
ARGS="${ARM_ARGS} ${CFG} --num_seconds=${SECS} --model_seed=101 --max_logs=60 \
--wandb ${ENTITY_ARG} --wandb_project=pinn_e5_smoke --wandb_name=${TAG}"

echo "=== E5-SMOKE task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} sys=${SYS} \
lr=${LR} damping=${DAMPING} momentum=${MOMENTUM} nc=${NC} secs=${SECS} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python -u train.py ${ARGS}
