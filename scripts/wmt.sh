#!/bin/bash
#
#SBATCH --job-name=wmt15
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/wyou/synst/experiments/wmt15/output_train.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wyou@cs.umass.edu

BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/wyou/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/wmt15

# Load in python3 and source the venv
module load python3/3.6.6-1810
source $PROJECT_PATH/../py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
PYTHONPATH=$BASE_PATH/wyou/py36/lib/python3.6/site-packages/:$PYTHONPATH

env $(cat ~/.comet.ml | xargs) python main.py --track -b 3175 --dataset wmt_en_de --span 1 \
  --model new_transformer --attn-param 1 --attn-type normal --attn-position left right --attn-weights 0 --attn-displacement 1 --embedding-size 512 --hidden-dim 2048 --num-heads 8 --num-layers 6 --attn-threshold 0.3 \
  --dec-attn-param 1 --dec-attn-type normal --dec-attn-position left center --dec-attn-weights 0 --dec-attn-displacement 1 --dec-attn-threshold 0.3 \
  --enc-dec-attn-param 1 --enc-dec-attn-type normal --enc-dec-attn-position left center right first --enc-dec-attn-score 1 --enc-dec-attn-displacement 1 --enc-dec-attn-threshold 0.3 \
  -d /mnt/nfs/work1/miyyer/wyou/wmt -p /mnt/nfs/work1/miyyer/wyou/wmt -v train \
  --checkpoint-interval 1200 --accumulate 2 --checkpoint-directory $EXPERIMENT_PATH


#env $(cat ~/.comet.ml | xargs) python $PROJECT_PATH/main.py -e 50 --experiment-path $EXPERIMENT_PATH/ --save-path checkpoint_512_l2d1_sp6_2500_a2_ld5_en_de_ \
#    --reverse -b $EXPERIMENT_PATH/model_best.pth.tar --accumulate-steps 2 --more-decoder-layers 1 --num-layers 2 --hidden-size 512 --minibatch-size 2500 --teacher-forcing-ratio 1 \
#    --clip 2.0 --dropout 0.3 --drop-last --rnn-type LSTM --track --batch-method token --span-size 6 --trim --filter --max-length 60 --lr-decay 1e-5 --num-evaluate 10 --learning-rate 0.001 --dataset IWSLT --optimizer Adam --num-directions 2 --lr-scheduler-type LambdaLR 
#    -r checkpoint_512_l4d1_sp4_9.pth.tar

#BASE_PARAMS=( \
#  -d "$BASE_PATH/datasets/wmt/" \
#  -p "$PROJECT_PATH/data/wmt" \
#  --dataset wmt_en_de_parsed \
#  --model parse_transformer \
#  --span 6 \
#  )
#
#env $(cat ~/.comet.ml | xargs) python main.py \
#  "${BASE_PARAMS[@]}" --batch-size 500 \
#  --restore $EXPERIMENT_PATH/checkpoint.pt --average-checkpoints 5 --split test \
#  translate --order-output --output-directory $EXPERIMENT_PATH
