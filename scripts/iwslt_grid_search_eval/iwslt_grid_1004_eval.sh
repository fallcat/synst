#!/bin/bash
#
#SBATCH --job-name=1004
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=47GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/iwslt_grid_1004/output_eval.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu
BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/iwslt_grid_1004

# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
#PYTHONPATH=$BASE_PATH/simengsun/synst/bin/lib/python3.6/site-packages/:$PYTHONPATH
PYTHONPATH=/mnt/nfs/work1/miyyer/wyou/py36/lib/python3.6/site-packages:$PYTHONPATH

	
CUDA_VISIBLE_DEVICES=0 python main.py --dataset iwslt_en_de --span 1   --model new_transformer \
  --attn-param 1 \
  --attn-type normal \
  --attn-position right first left left left left left left right first \
  --attn-displacement 1 \
  --dec-attn-param 1 \
  --dec-attn-type normal \
  --dec-attn-position center first left left left left left left center first \
  --dec-attn-displacement 1 \
  --embedding-size 286 --hidden-dim 507 --num-heads 2 --num-layers 5 \
  -d /mnt/nfs/work1/miyyer/wyou/iwslt -p /mnt/nfs/work1/miyyer/wyou/iwslt \
  --batch-size 1 --batch-method example --split dev \
  --restore $EXPERIMENT_PATH/checkpoint.pt \
  --average-checkpoints 5 \
  translate \
  --beam-width 4 --max-decode-length 50 --length-basis input_lens --order-output \
  --output-directory $EXPERIMENT_PATH
