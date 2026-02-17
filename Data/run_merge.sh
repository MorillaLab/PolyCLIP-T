#!/bin/bash
#SBATCH --job-name=mergeFam
# SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --partition=COMPUTE2
source ../polygeny/bin/activate


# python merge_family.py \
#    --family_id ROG \
#    --config family.json \
#    --out projects/ROG

# python merge_families.py \
#   --config families.json \
#   --out verif \
#   --all_families


python merge_family_new.py \
  --config families.json \
  --out result_test

# python build_clip_ml.py \
#   --tsv result_test/ALL_FAMILIES__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ./ml_data \
#   --test_family ROG
