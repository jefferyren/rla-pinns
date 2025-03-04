#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m3

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-1%1

echo "[DEBUG] Host name: " `hostname`

source  ~/miniforge3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 1 andresguzco/KFAC/s7i5sgpb