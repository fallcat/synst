#!/bin/bash
#
#SBATCH --job-name=wmt_enfr_left_eval
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/wmt_enfr_normal_right/output_eval.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu

BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/wmt_enfr_normal_right

# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
PYTHONPATH=$BASE_PATH/wyou/py36/lib/python3.6/site-packages/:$PYTHONPATH

#env $(cat ~/.comet.ml | xargs) python main.py --track -b 6000 --dataset iwslt_en_de --span 1 \
#  --embedding-size 286 --hidden-dim 507 --num-heads 2 --num-layers 5 \
#  -d /mnt/nfs/work1/miyyer/wyou/iwslt -p /mnt/nfs/work1/miyyer/wyou/iwslt -v train \
#  --checkpoint-interval 600 --accumulate 1 --learning-rate 3e-4 --checkpoint-directory /mnt/nfs/work1/miyyer/wyou/synst/experiments/iwslt01 \
#  --label-smoothing 0.0 --learning-rate-scheduler linear

CUDA_VISIBLE_DEVICES=0 python main.py --dataset wmt_en_fr --span 1 \
  --model new_transformer --attn-param 1 --attn-type normal --attn-position right --attn-displacement 1 --embedding-size 512 --hidden-dim 2048 --num-heads 8 --num-layers 6 \
  -d /mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr \
  -p /mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr \
  --batch-size 1 --batch-method example --split dev \
  --restore $EXPERIMENT_PATH/checkpoint.pt \
  --average-checkpoints 5 translate \
  --beam-width 4 --max-decode-length 50 --length-basis input_lens --order-output \
  --output-directory $EXPERIMENT_PATH


