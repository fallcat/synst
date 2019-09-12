
import os

# layer 123
enc_fixed = "left right left right left right"
dec_fixed = "left center left center left center"

enc_position_list = ['left left', 'left center', 'left right', 'center center', 'center right', 'right right']
dec_position_list = ['left left', 'left center', 'center center']

def generate_train_script(exp_id, enc_attn_pos, dec_attn_pos):
	s = """#!/bin/bash
#
#SBATCH --job-name=iwslt_grid_{0:04}
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=47GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/iwslt_grid_{0:04}/output_train.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu
BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/iwslt_grid_{0:04}\n
		""".format(exp_id)

	enc_p = enc_attn_pos.split()
	dec_p = dec_attn_pos.split()

	s += """
{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n
""".format(
		enc_p[0],enc_p[1], enc_p[-2], enc_p[-1],
		dec_p[0],dec_p[1], dec_p[-2], dec_p[-1]
	)

	s += """
# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
#PYTHONPATH=$BASE_PATH/simengsun/synst/bin/lib/python3.6/site-packages/:$PYTHONPATH
PYTHONPATH=/mnt/nfs/work1/miyyer/wyou/py36/lib/python3.6/site-packages:$PYTHONPATH\n
		"""
	s += """
env $(cat ~/.comet.ml | xargs) python main.py \\
  --track -b 6000 --dataset iwslt_en_de --span 1 \\
  --model new_transformer \\
  --attn-param 1 \\
  --attn-type normal \\
  --attn-position {0} \\
  --attn-displacement 1 \\
  --dec-attn-param 1 \\
  --dec-attn-type normal \\
  --dec-attn-position {1} \\
  --dec-attn-displacement 1 \\
  --embedding-size 286 --hidden-dim 507 --num-heads 2 --num-layers 5 \\
  -d /mnt/nfs/work1/miyyer/wyou/iwslt -p /mnt/nfs/work1/miyyer/wyou/iwslt -v train \\
  --checkpoint-interval 600 --accumulate 1 --learning-rate 3e-4 --checkpoint-directory $EXPERIMENT_PATH \\
  --label-smoothing 0.0 --learning-rate-scheduler linear\n
""".format(enc_attn_pos, dec_attn_pos)

	with open('./iwslt_grid_search/iwslt_grid_{0:04}.sh'.format(exp_id), 'w') as f:
		f.write(s)

def generate_eval_script(exp_id, enc_attn_pos, dec_attn_pos):
	s = """#!/bin/bash
#
#SBATCH --job-name=iwslt_grid_{0:04}
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=47GB
#SBATCH -d singleton
#SBATCH --open-mode append
#SBATCH -o /mnt/nfs/work1/miyyer/simengsun/synst/experiments/iwslt_grid_{0:04}/output_eval.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simengsun@cs.umass.edu
BASE_PATH=/mnt/nfs/work1/miyyer
PROJECT_PATH=$BASE_PATH/simengsun/synst
EXPERIMENT_PATH=$PROJECT_PATH/experiments/iwslt_grid_{0:04}\n
	""".format(exp_id)

	enc_p = enc_attn_pos.split()
	dec_p = dec_attn_pos.split()

	s += """
{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n
""".format(
		enc_p[0],enc_p[1], enc_p[-2], enc_p[-1],
		dec_p[0],dec_p[1], dec_p[-2], dec_p[-1]
	)

	s += """
# Load in python3 and source the venv
module load python3/3.6.6-1810
source /mnt/nfs/work1/miyyer/wyou/py36/bin/activate

# Need to include the venv path upfront...
# otherwise it defaults to the loaded slurm module which breaks for pynvml with python3
#PYTHONPATH=$BASE_PATH/simengsun/synst/bin/lib/python3.6/site-packages/:$PYTHONPATH
PYTHONPATH=/mnt/nfs/work1/miyyer/wyou/py36/lib/python3.6/site-packages:$PYTHONPATH\n
	"""
	s += """
CUDA_VISIBLE_DEVICES=0 python main.py --dataset iwslt_en_de --span 1 \
  --model new_transformer \\
  --attn-param 1 \\
  --attn-type normal \\
  --attn-position {0} \\
  --attn-displacement 1 \\
  --dec-attn-param 1 \\
  --dec-attn-type normal \\
  --dec-attn-position {1} \\
  --dec-attn-displacement 1 \\                     
  --embedding-size 286 --hidden-dim 507 --num-heads 2 --num-layers 5 \\
  -d /mnt/nfs/work1/miyyer/wyou/iwslt -p /mnt/nfs/work1/miyyer/wyou/iwslt \\
  --batch-size 1 --batch-method example --split dev \\                       
  --restore $EXPERIMENT_PATH/checkpoint.pt \\                               
  --average-checkpoints 5 \\
  translate \\               
  --beam-width 4 --max-decode-length 50 --length-basis input_lens --order-output \\
  --output-directory $EXPERIMENT_PATH
	""".format(enc_attn_pos, dec_attn_pos)

	with open('./iwslt_grid_search_eval/iwslt_grid_{0:04}_eval.sh'.format(exp_id), 'w') as f:
		f.write(s)

	if not os.path.exists('../experiments/iwslt_grid_{0:04}'.format(exp_id)):
		os.mkdir('../experiments/iwslt_grid_{0:04}'.format(exp_id))

exp_id = 0
for enc_l0_pos in enc_position_list:
	for enc_l4_pos in enc_position_list:
		for dec_l0_pos in dec_position_list:
			for dec_l4_pos in dec_position_list:
				enc_attn_pos = enc_l0_pos + " " + enc_fixed + " " + enc_l4_pos
				dec_attn_pos = dec_l0_pos + " " + dec_fixed + " " + dec_l4_pos
				generate_train_script(exp_id, enc_attn_pos, dec_attn_pos)
				generate_eval_script(exp_id, enc_attn_pos, dec_attn_pos)
				exp_id += 1

