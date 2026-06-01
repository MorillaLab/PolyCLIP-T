#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
##SBATCH --mem=700G
#SBATCH --partition=COMPUTE2
#SBATCH --output=log/evaluate.out
#SBATCH --error=log/evaluate.err

echo "========================================"
echo "Current hostname: $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "========================================"

source ~/polygeny/bin/activate



python final_evaluate.py
