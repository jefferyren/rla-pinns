#!/bin/bash
#SBATCH --job-name=e5_tune
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=00:50:00
#SBATCH --array=1-48
#SBATCH --output=logs/e5tune_%A_%a.out
#SBATCH --error=logs/e5tune_%A_%a.err
# =============================================================================
# E5(b) -- Tuning for PRIME-SR and Same-Sampled SPRING ONLY.
# See PINN_EXPERIMENTS.md section 4 / E5.
#
# ONE SYSTEM PER SUBMISSION. Pass it in, because each system has a different
# time budget and a shared --time would either truncate lfp9 or waste an hour
# of reservation on heat4:
#
#   sbatch --export=ALL,SYS=heat4 --time=00:25:00 runs/e5_tune.sh
#
# runs/e5_submit.sh does all four with the right --time. Use that.
# =============================================================================
#
# WHY ONLY TWO ARMS ARE TUNED
#   engdw and spring do not appear here at all. They run at the hyperparameters
#   published in arXiv:2505.12149 Appendix A.x.1, verbatim -- see e5_systems.sh.
#   Re-tuning them would replace a baseline the authors stand behind with one we
#   produced, and "we beat their method after retuning it ourselves" is a much
#   weaker claim than "we beat their method as published".
#
# THE TUNING ASYMMETRY, STATED PLAINLY
#   The paper gave each of its optimizers ~100 trials (50 wide + 50 refined) at
#   the FULL time budget. Each arm here gets 24 draws at 20% of the budget.
#   That is roughly an 8x smaller search, and it runs AGAINST primesr and
#   ssspring, not for them: our two arms are the under-tuned ones. Report it
#   that way. If they win anyway the result is stronger, not weaker; if they
#   lose, the honest reading is "not established at this search budget", and
#   the fix is more draws, not a different framing.
#
# PAIRED DESIGN
#   primesr and ssspring see the IDENTICAL (lr, damping) candidate list per
#   system. Neither method got a search the other did not. Say so in the paper,
#   because a shared list is a real design choice: it can disadvantage an arm
#   whose good region sits somewhere else entirely.
#
# DRAW 1 IS THE ANCHOR, not a random draw: it is the paper's own fixed-lr SPRING
# winner for that system, with norm_constraint 1e-3. It guarantees both arms are
# evaluated at a configuration already known to train that PDE, so a bad tuning
# result cannot be blamed on a search that never visited a workable region.
#
# DRAWS 2-24 are a fixed sample, python random.seed(20260906):
#   lr        LU([1e-4, 1e-1])   -- the paper's own fixed-lr search space
#   damping   LU per system      -- bracketing the paper's winners by ~2 decades
#                                   (p5 [1e-12,1e-5], p100 [1e-6,1e0],
#                                    heat4 [1e-9,1e-5], lfp9 [1e-6,1e-1])
#   momentum  U([0.8, 0.999])    -- the paper's fixed-lr SPRING momentum space.
#                                   ssspring only: this is its INITIAL beta,
#                                   which the controller then adapts.
#   norm_constraint  {1e-4, 1e-3, 1e-2}
#
#   PRIME-SR HAS NO MOMENTUM KNOB. It sets mu per step from the sampled Gram
#   matrix -- that is the entire method. So it tunes (lr, damping, nc) and
#   ssspring tunes (lr, damping, momentum, nc). The MOMENTA row is inert for
#   primesr and is still echoed in its header line, so the two arms' logs stay
#   parseable by one regex.
#
#   Values are HARD-CODED, not drawn at job time: the grid is then reproducible
#   from this file alone and the exact numbers can go straight into the
#   hyperparameter appendix.
#
# --model_seed=101 IS A DEDICATED TUNING SEED and must never appear in E5's
# reported figure. E5(c) reports seeds 1-3, held out from this selection.
#
# DRY RUN:
#   for i in $(seq 1 48); do SLURM_ARRAY_TASK_ID=$i SYS=p5 DRY_RUN=1 bash runs/e5_tune.sh; done
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
E5_REPO="${E5_REPO:-$HOME/rla-pinns-use}"
NDRAWS=24

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or for a dry run:" >&2
  echo "       SLURM_ARRAY_TASK_ID=1 SYS=p5 DRY_RUN=1 bash $0" >&2
  exit 1
fi
if [[ -z "${SYS:-}" ]]; then
  echo "ERROR: SYS unset. This script tunes ONE system per submission:" >&2
  echo "       sbatch --export=ALL,SYS=p5 --time=00:35:00 runs/e5_tune.sh" >&2
  echo "       (or just use runs/e5_submit.sh tune)" >&2
  exit 1
fi

SYSTEMS_FILE="${E5_REPO}/runs/e5_systems.sh"
[[ -f "$SYSTEMS_FILE" ]] || SYSTEMS_FILE="$(cd "$(dirname "$0")" && pwd)/e5_systems.sh"
[[ -f "$SYSTEMS_FILE" ]] || { echo "ERROR: cannot find e5_systems.sh" >&2; exit 1; }
# shellcheck source=/dev/null
source "$SYSTEMS_FILE"

e5_sys_config "$SYS"

# --- Candidate sets: draw 1 = paper's SPRING winner, draws 2-24 fixed random --
LRS_P5=(0.063502 0.0461925 0.00581559 0.0646934 0.000793374 0.000471534 0.048501 0.0702158 0.000363774 0.004629 0.0453538 0.00264245 0.000334509 0.035215 0.0163656 0.000180584 0.00743404 0.000376822 0.00243716 0.0492919 0.00741517 0.000115387 0.000100961 0.00789686)
DAMPINGS_P5=(6.81158e-10 1.39095e-10 5.99033e-11 2.45748e-11 6.74232e-12 7.53745e-10 9.25046e-08 2.41609e-08 4.80695e-11 2.66824e-11 7.17827e-09 8.42938e-07 2.49991e-12 1.36368e-06 3.35908e-08 6.0582e-12 9.20889e-08 1.523e-11 4.72631e-10 2.41844e-07 1.82173e-09 4.95765e-06 3.74565e-11 3.23857e-10)
MOMENTA_P5=(0.826966 0.916357 0.917511 0.971144 0.934764 0.875153 0.815679 0.905236 0.968329 0.808021 0.995429 0.896273 0.959945 0.925778 0.826981 0.830435 0.900523 0.950689 0.961809 0.830208 0.868301 0.808709 0.885964 0.980518)
NCS_P5=(0.001 0 0 0 0.1 0 10 0 1000 0.001 0.001 0 0 0.001 0.001 0.1 10 10 10 0 0 1000 1000 0.001)

LRS_P100=(0.092362 0.0672031 0.00236619 0.0548919 0.000153235 0.0917071 0.0005978 0.00378986 0.000746762 0.00319174 0.0687118 0.00479284 0.00347303 0.0013879 0.0242176 0.0620526 0.00623199 0.0346698 0.00254071 0.0415626 0.000175675 0.0365123 0.00113339 0.000127271)
DAMPINGS_P100=(0.030116 0.192576 3.37122e-05 2.00057e-06 0.00238863 0.126872 0.00174262 3.24429e-05 2.1991e-06 5.41649e-05 0.0223902 0.0398148 0.101205 2.82694e-05 0.0031716 0.016838 1.05477e-05 0.0509123 0.00318609 0.00291988 0.00629884 0.000260346 1.44677e-06 0.00106748)
MOMENTA_P100=(0.98386 0.888614 0.973567 0.982828 0.898955 0.821563 0.965917 0.818862 0.830645 0.880357 0.816507 0.861768 0.955621 0.969766 0.895733 0.951331 0.998671 0.961952 0.817743 0.849918 0.851686 0.962654 0.976288 0.962437)
NCS_P100=(0.001 10 0 0 10 10 0.1 0 1000 0.1 0 10 0.1 10 0 0.001 0.001 0 0 0 0 10 1000 0.1)

LRS_HEAT4=(0.0866389 0.0245968 0.0281891 0.00065188 0.000559922 0.0145016 0.00845918 0.000374981 0.0164973 0.00213521 0.023848 0.00260424 0.000659963 0.0116396 0.0724414 0.000146642 0.000774316 0.000114239 0.000511302 0.000143894 0.0720897 0.000555929 0.000301069 0.0442781)
DAMPINGS_HEAT4=(2.08178e-07 8.72296e-08 1.44051e-06 3.02809e-08 1.16491e-09 9.36948e-07 3.69387e-06 5.36784e-09 2.33648e-07 3.75285e-07 7.69223e-06 4.53557e-06 3.67345e-08 1.4697e-06 1.83266e-08 1.86131e-07 8.42295e-07 5.90592e-08 1.13132e-07 3.67635e-06 1.54133e-06 4.34367e-08 8.12556e-08 4.11563e-08)
MOMENTA_HEAT4=(0.907846 0.943371 0.920669 0.967171 0.824078 0.809402 0.955999 0.928861 0.842777 0.969224 0.829841 0.960586 0.888772 0.899871 0.91687 0.997731 0.933187 0.815625 0.907705 0.910716 0.947216 0.93506 0.893492 0.87851)
NCS_HEAT4=(0.001 0.1 0.1 10 0.001 0 0 0.001 0 0 0 0 0 0.1 10 0 10 0.1 0.001 0.1 1000 10 0 0)

LRS_LFP9=(0.0447319 0.00204901 0.000352645 0.000375971 0.00024771 0.000430057 0.000780288 0.00233193 0.00259378 0.00455255 0.000261257 0.00175656 0.0617084 0.00161406 0.000301383 0.0559364 0.00879042 0.0171692 0.0383897 0.00436438 0.0312438 0.00120642 0.00437921 0.0002574)
DAMPINGS_LFP9=(0.00837765 0.000210215 0.00074081 0.000124327 2.65711e-06 0.00295177 0.0103984 0.0571157 0.000257751 5.99017e-05 0.000121156 0.00325346 0.0125831 0.00123066 0.000471825 0.00500111 0.000301593 7.26924e-06 0.000214062 0.00248449 2.69156e-06 0.0583569 0.00037923 0.0964806)
MOMENTA_LFP9=(0.976009 0.807559 0.800871 0.931599 0.934353 0.880119 0.991541 0.877458 0.902974 0.873398 0.864195 0.958568 0.976315 0.845733 0.876275 0.901801 0.884494 0.828548 0.986047 0.926476 0.802772 0.912667 0.811429 0.994567)
NCS_LFP9=(0.001 0 1000 1000 0.001 1000 1000 0 10 0 1000 0.1 0.1 0 0.001 1000 0 0.1 10 0 1000 0.1 0.1 0.001)

UP=$(echo "$SYS" | tr '[:lower:]' '[:upper:]')
eval "LRS=(\"\${LRS_${UP}[@]}\")"
eval "DAMPINGS=(\"\${DAMPINGS_${UP}[@]}\")"
eval "MOMENTA=(\"\${MOMENTA_${UP}[@]}\")"
eval "NCS=(\"\${NCS_${UP}[@]}\")"

if (( ${#LRS[@]} != NDRAWS )); then
  echo "ERROR: ${SYS} has ${#LRS[@]} lr draws, expected ${NDRAWS}" >&2
  exit 1
fi

# --- Grid: 24 draws x 2 arms = 48 --------------------------------------------
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
draw_idx=$(( i % NDRAWS ))
arm_idx=$(( i / NDRAWS ))
if (( arm_idx > 1 )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside 1-48" >&2
  exit 1
fi

TUNE_ARMS=(primesr ssspring)
ARM=${TUNE_ARMS[$arm_idx]}
LR=${LRS[$draw_idx]}
DAMPING=${DAMPINGS[$draw_idx]}
MOMENTUM=${MOMENTA[$draw_idx]}
NC=${NCS[$draw_idx]}
DRAW=$(( draw_idx + 1 ))

if [[ "$DRY_RUN" != "1" ]]; then
  e5_activate
fi
e5_entity_arg
e5_arm_args "$ARM" "$LR" "$DAMPING" "$MOMENTUM" "$NC"

TAG="${ARM}_${SYS}_d${DRAW}"
ARGS="${ARM_ARGS} ${CFG} --num_seconds=${TUNE_BUDGET} --model_seed=101 --max_logs=100 \
--wandb ${ENTITY_ARG} --wandb_project=pinn_e5_tune --wandb_name=${TAG}"

echo "=== E5-TUNE task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} sys=${SYS} \
draw=${DRAW}/${NDRAWS} lr=${LR} damping=${DAMPING} momentum=${MOMENTUM} nc=${NC} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python -u train.py ${ARGS}
