#!/bin/bash
#
#SBATCH --job-name=wmt_enfr_learned_eval
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/wmt_enfr_learned/output_eval.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu

BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/wmt_enfr_learned

# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
PYTHONPATH=$BASE_PATH/wyou/py36/lib/python3.6/site-packages/:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0 python main.py --dataset wmt_en_fr --span 1 \
  --model new_transformer --attn-type learned\
  -d /mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr \
  -p /mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr \
  --batch-size 1 --batch-method example --split test \
  --restore $EXPERIMENT_PATH/checkpoint.pt \
  --average-checkpoints 5 translate \
  --beam-width 4 --max-decode-length 50 --length-basis input_lens --order-output \
  --output-directory $EXPERIMENT_PATH


