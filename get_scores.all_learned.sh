#!/bin/bash
#
#SBATCH --job-name=scores
#SBATCH --partition=titanx-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=47GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH --output=/mnt/nfs/work1/miyyer/simengsun/synst/lingeval/get_scores.out
#SBATCH --error=/mnt/nfs/work1/miyyer/simengsun/synst/lingeval/get_scores.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu

python get_scores.py --dataset wmt_en_de --span 1 \
  --model new_transformer \
  --attn-param 1 --attn-type learned \
  --attn-position left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right --attn-displacement 1 --embedding-size 512 --hidden-dim 2048 --num-heads 8 --num-layers 6 \
  --dec-attn-param 1 --dec-attn-type learned --dec-attn-position left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center left center --dec-attn-displacement 1 \
  --enc-dec-attn-type learned \
  -d /mnt/nfs/work1/miyyer/wyou/wmt -p /mnt/nfs/work1/miyyer/wyou/wmt \
  --batch-size 1 --batch-method example --split test \
  --restore $EXPERIMENT_PATH/checkpoint.pt \
  --average-checkpoints 5 evaluate
