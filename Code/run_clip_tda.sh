#!/bin/bash
#SBATCH --job-name=tda
#SBATCH --nodes=1
##SBATCH --mem=1500G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=SMP-256c
#SBATCH --output=polyclip_tda.out
#SBATCH --error=polyclip_tda.err

# Activate virtual environment
source ../../polygeny/bin/activate

echo "========================================"
echo "Current hostname: $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "========================================"

# Set up environment variables
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

# Get master address
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=29503

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# --nnodes=$SLURM_NNODES \

srun --jobid $SLURM_JOBID bash -c \
  'torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$SLURM_PROCID \
    trainning.py --mode clip_tda_labelled --run_name clip_tda_labelled_parallel'

echo "========================================"
echo "Training completed!"
echo "========================================"



