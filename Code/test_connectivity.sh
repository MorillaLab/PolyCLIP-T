#!/bin/bash
#SBATCH --job-name=s_text_onto
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
##SBATCH --mem=500G
#SBATCH --partition=COMPUTE2
#SBATCH --output=log/s_onto.out
#SBATCH --error=log/s_onto.err

##SBATCH --job-name=new_ref_v2
##SBATCH --nodes=7
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=384
##SBATCH --mem=700G
##SBATCH --partition=COMPUTE2
##SBATCH --output=log/train.out
##SBATCH --error=log/train.err

echo "========================================"
echo "Current hostname: $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "========================================"

source ~/polygeny/bin/activate

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

# MASTER NODE
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=29550

export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

echo "========================================"
echo "Starting distributed training"
echo "========================================"


srun --jobid $SLURM_JOBID bash -c \
  'torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$SLURM_PROCID \
    trainning.py'

echo "========================================"
echo "Training completed!"
echo "========================================"