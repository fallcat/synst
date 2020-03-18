#!/bin/bash
#
#SBATCH --job-name=enfr_04
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/wmt_enfr_04/output_train.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu

BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/wmt_enfr_04

# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
PYTHONPATH=$BASE_PATH/wyou/py36/lib/python3.6/site-packages/:$PYTHONPATH

env $(cat ~/.comet.ml | xargs) python main.py --track -b 3175 --dataset wmt_en_fr --span 1 \
  --model new_transformer \
  --attn-param 1 \
  --attn-type normal \
  --attn-position left right left right left right left right left right \
  --attn-displacement 1 \
  --dec-attn-param 1 \
  --dec-attn-type normal \
  --dec-attn-position left center left center left center left center left center \
  --dec-attn-displacement 1 \
  --enc-dec-attn-type learned \
  --embedding-size 512 --hidden-dim 2048 --num-heads 8 --num-layers 6 \
  -d /mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr \
  -p /mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr \
  -v train \
  --checkpoint-interval 1200 --accumulate 2 --checkpoint-directory $EXPERIMENT_PATH