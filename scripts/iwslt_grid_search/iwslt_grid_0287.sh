#!/bin/bash
#
#SBATCH --job-name=iwslt_grid_0287
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=47GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/iwslt_grid_0287/output_train.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu
BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/iwslt_grid_0287

		
right	right	left	center	center	center	center	center	


# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
#PYTHONPATH=$BASE_PATH/simengsun/synst/bin/lib/python3.6/site-packages/:$PYTHONPATH
PYTHONPATH=/mnt/nfs/work1/miyyer/wyou/py36/lib/python3.6/site-packages:$PYTHONPATH

		
env $(cat ~/.comet.ml | xargs) python main.py \
  --track -b 6000 --dataset iwslt_en_de --span 1 \
  --model new_transformer \
  --attn-param 1 \
  --attn-type normal \
  --attn-position right right left right left right left right left center \
  --attn-displacement 1 \
  --dec-attn-param 1 \
  --dec-attn-type normal \
  --dec-attn-position center center left center left center left center center center \
  --dec-attn-displacement 1 \
  --embedding-size 286 --hidden-dim 507 --num-heads 2 --num-layers 5 \
  -d /mnt/nfs/work1/miyyer/wyou/iwslt -p /mnt/nfs/work1/miyyer/wyou/iwslt -v train \
  --checkpoint-interval 600 --accumulate 1 --learning-rate 3e-4 --checkpoint-directory $EXPERIMENT_PATH \
  --label-smoothing 0.0 --learning-rate-scheduler linear

