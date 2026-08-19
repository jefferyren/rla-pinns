#!/bin/bash
#SBATCH --job-name=e2_tune
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=00:40:00
#SBATCH --array=1-78
#SBATCH --output=logs/e2_%A_%a.out
#SBATCH --error=logs/e2_%A_%a.err
# =============================================================================
# E2 -- Equal-budget tuning.  See PINN_EXPERIMENTS.md section 4 / E2.
# =============================================================================
#
# THIS BLOCK DOES DOUBLE DUTY. It is both the tuning that makes Tab A's baseline
# honest, and the entire evidence for C1 ("Adaptive SPRING fails, and not because
# it was under-tuned").
#
# WHY IT EXISTS
#   Every wandb sweep in this repo sets metric:{goal:minimize, name:l2_error} --
#   hyperparameters chosen on THE NUMBER THE PAPER REPORTS, from THE SAME RUNS
#   THE PAPER REPORTS -- and the budgets were wildly unequal: the RNGD/ENGD/KFAC/
#   Adam/SGD baselines got ~50 random draws each across 180 YAMLs, while every
#   SPRING-family run used ONE hand-picked pair inherited from a different
#   optimizer's winner. As drafted, "Adaptive SPRING fails" reads as "Adaptive
#   SPRING was never tuned." This block removes that reading.
#
# DESIGN
#   13 candidate (lr, damping, momentum) triples x 3 arms x 2 PDEs = 78 tasks.
#   Draw 1 is the inherited anchor (0.0924, 0.0301, 0.99) so the historical
#   configuration is represented in the candidate set. Draws 2-13 are a fixed
#   random sample: lr log-uniform [3e-3, 3e-1], damping log-uniform [1e-5, 1e-1],
#   momentum uniform [0.9, 0.999], python random.seed(20260819).
#
#   The triples are HARD-CODED rather than drawn at job time, for three reasons:
#   the grid is reproducible from this file alone, the exact values can go
#   straight into app:hyperparams, and every arm is evaluated on the IDENTICAL
#   candidate set (a paired design -- tighter than independent draws per arm, and
#   it forecloses "you searched harder for your own method"). State in the paper
#   that the candidate set was shared, since it is a real design choice: a shared
#   set could in principle disadvantage an arm whose good region lies elsewhere.
#
#   --model_seed=101 is a DEDICATED TUNING SEED and must never appear in Fig A or
#   Tab A. This is what makes the selection honest: the reported runs (E3, seeds
#   1-3) are held out from the selection that produced their hyperparameters.
#
# WHAT TO REPORT
#   Not just the winner. Report the full 13-draw spread per arm (min/median/max
#   final l2_error) -- that scatter IS the C1 evidence, not merely a selection
#   step. runs/harvest_e2.py emits both the winners and the spread.
#
# KILL CRITERION
#   If some draw makes Adaptive SPRING match tuned SPRING on BOTH PDEs, C1 is
#   dead as stated. The honest reframing is then that the controller is
#   hyperparameter-FRAGILE rather than broken, SS-SPRING's claim becomes
#   robustness rather than rescue, and the abstract and title must change.
#
# DRY RUN -- verify all 78 command lines without touching the cluster:
#   for i in $(seq 1 78); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash runs/e2_tune.sh; done
# =============================================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID unset. Submit with sbatch, or for a dry run:" >&2
  echo "       SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash $0" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
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

# --- Candidate set: draw 1 = inherited anchor, draws 2-13 = fixed random sample
LRS=(0.0924 0.028552 0.0128528 0.0168818 0.115177 0.126093 0.183333 \
0.00739935 0.0122429 0.058334 0.0189255 0.0163274 0.141084)
DAMPINGS=(0.0301 0.00139252 0.000434206 1.32294e-05 0.00157107 1.20481e-05 \
0.000130672 1.05784e-05 0.00144415 0.0603175 0.00285587 0.0110596 0.00180014)
MOMENTA=(0.99 0.96133 0.96916 0.91135 0.92724 0.99304 0.94150 0.97985 0.92923 \
0.91247 0.94563 0.94452 0.90879)
NDRAWS=13

P100="--equation=poisson --boundary_condition=u_weinan_norm --dim_Omega=100 \
--model=mlp-tanh-768-768-512-512 --N_Omega=200 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"

LFP9="--equation=log-fokker-planck-isotropic --boundary_condition=gaussian \
--dim_Omega=9 --model=mlp-tanh-256-256-128-128 --N_Omega=300 --N_dOmega=100 \
--N_eval=30000 --dtype=float64 --batch_frequency=1"

# --- Grid: 13 draws x 2 PDEs x 3 arms = 78 -----------------------------------
i=$(( SLURM_ARRAY_TASK_ID - 1 ))
draw_idx=$(( i % NDRAWS ))
pde_idx=$(( (i / NDRAWS) % 2 ))
arm_idx=$(( i / (NDRAWS * 2) ))

ARMS=(a1_spring a2_adaptive a3_ss)
ARM=${ARMS[$arm_idx]}
LR=${LRS[$draw_idx]}
DAMPING=${DAMPINGS[$draw_idx]}
MOMENTUM=${MOMENTA[$draw_idx]}
DRAW=$(( draw_idx + 1 ))

if [[ $pde_idx -eq 0 ]]; then PDE=p100; CFG="$P100"; else PDE=lfp9; CFG="$LFP9"; fi

TAG="${ARM}_${PDE}_d${DRAW}"
COMMON="${CFG} --num_seconds=1200 --model_seed=101 --max_logs=150 \
--wandb --wandb_entity=rla-pinns --wandb_project=pinn_e2_tune --wandb_name=${TAG}"

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

echo "=== E2 task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} pde=${PDE} draw=${DRAW}/13 \
lr=${LR} damping=${DAMPING} momentum=${MOMENTUM} ==="
echo "ARGS: ${ARGS}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run -- not executing)"
  exit 0
fi

python -u train.py ${ARGS}
