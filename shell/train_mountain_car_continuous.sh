#!/bin/bash
#SBATCH --job-name=train_mountain_car_continuous
#SBATCH --time=20:00:00
#SBATCH --output=../slurm_logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1

singularity exec --nv ../rl.sif python3 ../python/train_mountain_car_continuous.py