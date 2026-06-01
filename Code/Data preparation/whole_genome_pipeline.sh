#!/bin/bash
#SBATCH --job-name=mergeFam
#SBATCH --partition=SMP-256c
##SBATCH --nodelist=magi170
#SBATCH --mem=1200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --array=1-5
##SBATCH --array=2
##SBATCH --output=logs/%x_%A_%a.out
##SBATCH --error=logs/%x_%A_%a.err
##SBATCH --output=logs/res2.out
#SBATCH --output=logs/res3.out
#SBATCH --error=logs/res3.err

set -euo pipefail
source ~/polygeny/bin/activate

FAMILY_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" family_ids.txt)
echo "Running family: ${FAMILY_ID}"

# python whole_genome_family_pipeline.py \
#   --config families.json \
#   --out_dir Genome_Data_Final

python whole_genome_family_pipeline.py \
  --config families.json \
  --out_dir Genome_Data_Final \
  --family_id "${FAMILY_ID}"
