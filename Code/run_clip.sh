#!/bin/bash
#SBATCH --job-name=clip
#SBATCH --nodes=9
#SBATCH --mem=600G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=COMPUTE2
#SBATCH --output=clip.out
#SBATCH --error=clip.err

# Activate virtual environment
source ../../polygeny/bin/activate

echo "========================================"
echo "Current hostname: $(hostname)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "========================================"

# Set up environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

# Get master address
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=29501

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
    trainning.py --mode clip_only --run_name clip_only_parallel'

echo "========================================"
echo "Training completed!"
echo "========================================"














# #!/bin/bash
# #SBATCH --job-name=refalt_3parallel
# #SBATCH --nodes=10
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=5
# #SBATCH --partition=MISC-56c-VERYSHORT 
# #SBATCH --output=train_3parallel_%j.out
# #SBATCH --error=train_3parallel_%j.err

# source ../../polygeny/bin/activate

# export CUDA_VISIBLE_DEVICES=""
# export PYTHONUNBUFFERED=1
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# # ---- build a host array from the allocated nodes
# mapfile -t HOSTS < <(scontrol show hostname "$SLURM_NODELIST")
# echo "Allocated nodes: ${#HOSTS[@]}"
# printf "%s\n" "${HOSTS[@]}"

# # ---- split nodes: 4 / 3 / 3
# NODES_A=("${HOSTS[@]:0:4}")    # clip_only
# NODES_B=("${HOSTS[@]:4:3}")    # labelled_clip
# NODES_C=("${HOSTS[@]:7:3}")    # clip_tda_labelled

# NODELIST_A=$(IFS=, ; echo "${NODES_A[*]}")
# NODELIST_B=$(IFS=, ; echo "${NODES_B[*]}")
# NODELIST_C=$(IFS=, ; echo "${NODES_C[*]}")

# MASTER_A=${NODES_A[0]}
# MASTER_B=${NODES_B[0]}
# MASTER_C=${NODES_C[0]}

# # ---- per-run ports (must be different)
# PORT_A=29501
# PORT_B=29502
# PORT_C=29503

# # ---- processes per node (CPU DDP)
# NPROC_PER_NODE=3

# echo "RUN A nodes: $NODELIST_A | master=$MASTER_A:$PORT_A"
# echo "RUN B nodes: $NODELIST_B | master=$MASTER_B:$PORT_B"
# echo "RUN C nodes: $NODELIST_C | master=$MASTER_C:$PORT_C"


# srun --exclusive --nodes=4 --ntasks=4 --ntasks-per-node=1 -w "$NODELIST_A" \
#   bash -c "torchrun --nnodes=4 --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=$MASTER_A --master_port=$PORT_A --node_rank=\$SLURM_NODEID \
#     trainning.py --mode clip_only --run_name clip_only_parallel" &

# srun --exclusive --nodes=3 --ntasks=3 --ntasks-per-node=1 -w "$NODELIST_B" \
#   bash -c "torchrun --nnodes=3 --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=$MASTER_B --master_port=$PORT_B --node_rank=\$SLURM_NODEID \
#     trainning.py --mode labelled_clip --run_name labelled_clip_parallel" &

# srun --exclusive --nodes=3 --ntasks=3 --ntasks-per-node=1 -w "$NODELIST_C" \
#   bash -c "torchrun --nnodes=3 --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=$MASTER_C --master_port=$PORT_C --node_rank=\$SLURM_NODEID \
#     trainning.py --mode clip_tda_labelled --run_name clip_tda_labelled_parallel" &

# wait
# echo "All 3 runs finished."
