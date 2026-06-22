#!/bin/bash
#SBATCH --job-name=pinns_test
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=03:00:00
#SBATCH --array=1-12
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err

# --- Environment setup ---
# Batch jobs do NOT read .bashrc by default (your interactive `bash -i` did),
# so conda must be initialized explicitly here or `conda activate` will fail.
source ~/.bashrc
conda activate rla_pinns || { echo "ERROR: conda activate failed"; exit 1; }

cd ~/rla-pinns-use/rla_pinns || { echo "ERROR: cd failed"; exit 1; }

# --- One full command per array task ---
case $SLURM_ARRAY_TASK_ID in
  1) ARGS="--wandb --wandb_name=primesr_N20 --optimizer=PRIMESR --PRIMESR_lr=0.0924 --PRIMESR_damping=0.0301 --PRIMESR_norm_constraint=1e-3 --PRIMESR_print_every=100 --N_Omega=20 --N_dOmega=10 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  2) ARGS="--wandb --wandb_name=primesr_N50 --optimizer=PRIMESR --PRIMESR_lr=0.0924 --PRIMESR_damping=0.0301 --PRIMESR_norm_constraint=1e-3 --PRIMESR_print_every=100 --N_Omega=50 --N_dOmega=25 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  3) ARGS="--wandb --wandb_name=primesr_N100 --optimizer=PRIMESR --PRIMESR_lr=0.0924 --PRIMESR_damping=0.0301 --PRIMESR_norm_constraint=1e-3 --PRIMESR_print_every=100 --N_Omega=100 --N_dOmega=50 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  4) ARGS="--wandb --wandb_name=primesr_N200 --optimizer=PRIMESR --PRIMESR_lr=0.0924 --PRIMESR_damping=0.0301 --PRIMESR_norm_constraint=1e-3 --PRIMESR_print_every=100 --N_Omega=200 --N_dOmega=100 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  5) ARGS="--wandb --wandb_name=ss_spring_unified_N20 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --N_Omega=20 --N_dOmega=10 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  6) ARGS="--wandb --wandb_name=ss_spring_unified_N50 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --N_Omega=50 --N_dOmega=25 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  7) ARGS="--wandb --wandb_name=ss_spring_unified_N100 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --N_Omega=100 --N_dOmega=50 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  8) ARGS="--wandb --wandb_name=ss_spring_unified_N200 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --N_Omega=200 --N_dOmega=100 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  9) ARGS="--wandb --wandb_name=ss_spring_unified_adaptive_N20 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --SameSampledSPRINGUnified_adaptive_eta --N_Omega=20 --N_dOmega=10 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  10) ARGS="--wandb --wandb_name=ss_spring_unified_adaptive_N50 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --SameSampledSPRINGUnified_adaptive_eta --N_Omega=50 --N_dOmega=25 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  11) ARGS="--wandb --wandb_name=ss_spring_unified_adaptive_N100 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --SameSampledSPRINGUnified_adaptive_eta --N_Omega=100 --N_dOmega=50 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  12) ARGS="--wandb --wandb_name=ss_spring_unified_adaptive_N200 --optimizer=SameSampledSPRINGUnified --SameSampledSPRINGUnified_lr=0.0924 --SameSampledSPRINGUnified_damping=0.0301 --SameSampledSPRINGUnified_momentum=0 --SameSampledSPRINGUnified_lb_window=30 --SameSampledSPRINGUnified_probe_lr=0.0924 --SameSampledSPRINGUnified_probe_damping=0.0301 --SameSampledSPRINGUnified_adaptive_eta --N_Omega=200 --N_dOmega=100 --batch_frequency=1 --boundary_condition=u_weinan_norm --dim_Omega=100 --equation=poisson --model=mlp-tanh-768-768-512-512 --num_seconds=7000" ;;
  *) echo "ERROR: no command defined for task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID running: python train.py $ARGS"
python train.py $ARGS
