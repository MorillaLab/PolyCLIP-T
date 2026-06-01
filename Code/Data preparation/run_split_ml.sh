#!/bin/bash
#SBATCH --job-name=mergeFam
#SBATCH --partition=COMPUTE2
##SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --output=logs/split_ml.out
#SBATCH --error=logs/split_ml.err

source ~/polygeny/bin/activate

#Random eval 500 (keeps train unchanged)

# python build_clip_ml.py \
#   --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA_NEW \
#   --test_family AGZ \
#   --train_n 5000 \
#   --train_seed 42 \
#   --eval_n 500 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

  python build_clip_ml.py \
  --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
  --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
  --out_dir ML_DATA_NEW/ROG \
  --test_family ROG \
  --train_n 5000 \
  --train_seed 42 \
  --eval_n 500 \
  --eval_seed 7 \
  --eval_mode random \
  --export_text \
  --export_onto \
  --export_textfields

  python build_clip_ml.py \
  --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
  --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
  --out_dir ML_DATA_NEW/DIA \
  --test_family DIA \
  --train_n 5000 \
  --train_seed 42 \
  --eval_n 500 \
  --eval_seed 7 \
  --eval_mode random \
  --export_text \
  --export_onto \
  --export_textfields

  python build_clip_ml.py \
  --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
  --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
  --out_dir ML_DATA_NEW/MER \
  --test_family MER \
  --train_n 5000 \
  --train_seed 42 \
  --eval_n 500 \
  --eval_seed 7 \
  --eval_mode random \
  --export_text \
  --export_onto \
  --export_textfields

  python build_clip_ml.py \
  --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
  --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
  --out_dir ML_DATA_NEW/CIS \
  --test_family CIS \
  --train_n 5000 \
  --train_seed 42 \
  --eval_n 500 \
  --eval_seed 7 \
  --eval_mode random \
  --export_text \
  --export_onto \
  --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA_NEW \
#   --test_family AGZ \
#   --train_n 5000 \
#   --train_seed 42 \
#   --eval_n 500 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA_NEW \
#   --test_family AGZ \
#   --eval_n 5000 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/ALL_FAMILY_FAMILY_MAX_POP_UPDATE.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA \
#   --test_family AGZ \
#   --eval_n 500 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields


# python build_clip_ml.py \
#   --tsv Exome_Data/ALL_FAMILIES__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA \
#   --test_family AGZ \
#   --eval_n 500 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Genome_Data/ALL_FAMILIES__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA_GENOME \
#   --test_family AGZ \
#   --eval_n 500 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/ALL_FAMILIES__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA \
#   --test_family MER \
#   --eval_n 0 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields


# python build_clip_ml.py \
#   --tsv Exome_Data/AGZ/AGZ__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA/Evaluation/AGZ --test_family DUMMY \
#   --eval_n 0 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/CIS/CIS__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA/Evaluation/CIS --test_family DUMMY \
#   --eval_n 0 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/DIA/DIA__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA/Evaluation/DIA --test_family DUMMY \
#   --eval_n 0 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/MER/MER__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA/Evaluation/MER --test_family DUMMY \
#   --eval_n 0 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

# python build_clip_ml.py \
#   --tsv Exome_Data/ROG/ROG__FAMILY_VARIANTS.tsv \
#   --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa \
#   --out_dir ML_DATA/Evaluation/ROG --test_family DUMMY \
#   --eval_n 0 \
#   --eval_seed 7 \
#   --eval_mode random \
#   --export_text \
#   --export_onto \
#   --export_textfields

#Stratified by labels + remove eval variants from training
# python build_clip_ml.py --tsv /nfs/homes/vomodonfack/Dataset/Exome_Data/ALL_FAMILIES__GLOBAL_UNIQUE_VARIANTS1.tsv --fasta /nfs/homes/vomodonfack/these/MER/dataset/hg38.fa --out_dir ML_DATA --test_family DUMMY \
#   --eval_n 500 --eval_seed 7 --eval_mode stratified_label --eval_exclude_from_train