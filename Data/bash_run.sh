#!/bin/bash
#SBATCH --job-name=split_data
# SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --partition=COMPUTE2
source ../../polygeny/bin/activate

# python prepare_ml_global_unique.py \
#   --tsv ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv \
#   --out_dir ml_ready_global_unique \
#   --label label_of_interest \
#   --test_frac 0.2 \
#   --seed 13

# python build_clip_ml.py \
#   --tsv result_test/ALL_FAMILIES__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ./ml_data \
#   --test_family ROG

python prepare_polyclip_views_and_split.py \
      --tsv ../result_test/ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS.tsv \
      --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
      --out_dir ../clip_ready \
      --half_window 50 \
      --label label_of_interest \
      --val_frac 0.2 \
      --seed 13
