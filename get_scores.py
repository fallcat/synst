
"""
	
	Get cross_entropy loss per token of sentences

	Input: 

	1. the binarized batch object stored in 
		/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.synst.32000.pkl

		{'batches': batches}, batches is a list of dictionary:
			{'example_ids': (0,), 
			'inputs': tensor([[ 9846,  3826, 39, 24238,  8223,  1308, 14, 30483,   962]]), 
			'input_lens': tensor([9]), 
			'targets': tensor([[37006,  9846, 3826, 39,  8545, 189, 846, 19, 30928,  1045, 37007]]), 
			'target_lens': tensor([11])}
	
	2. model: all learned / only learn source /learn 2 self-attns


	Output: scores of each sentence, save to 
		/mnt/nfs/work1/miyyer/simengsun/lingeval97/synst.{model-type}.scores

"""
import os
import pdb
import torch
import pickle
from args import parse_args
from datetime import datetime
from models.utils import restore
from binarize_lingeval97 import load_vocab

LINGEVAL97_PATH = "/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.synst.32000.pkl"
SCORES_PATH = "/mnt/nfs/work1/miyyer/simengsun/lingeval97/synst"

def get_scores(model, batches):

	scores = []
	with torch.no_grad():
		for idx, batch in enumerate(batches):
			_, nll = model(batch)
			scores.append(nll.item() / torch.sum(batch['target_lens']).item())
			if idx % 500 == 0:
				print("{} end processing {} sentences".format(datetime.now(), idx+1))
	return scores

class DummyDataset(object):

	def __init__(self, t2i, i2t):
		self.vocab_size = len(i2t)
		self.padding_idx = t2i['<PAD>']
		self.sos_idx = t2i['<SOS>']
		self.word_count_ratio = float(4215814/4186988) 
		self.word_align_stats = None

if __name__ == "__main__":

	# args
	print("parsing args ...")
	args = parse_args(argv=None)

	score_fname = os.path.dirname(args.restore)
	score_fname = os.path.basename(score_fname)
	print("Will store score file to %s" % (os.path.join(SCORES_PATH, score_fname)))

	# load dummy dataset, while init model, only need dataset.vocab_size/padding_idx/sos_idx
	print("loading vocab ...")
	t2i, i2t = load_vocab()
	dummy_dataset = DummyDataset(t2i, i2t)

	# load lingeval97 data
	print("loading dataset ...")
	with open(LINGEVAL97_PATH, 'rb') as f:
		data = pickle.load(f)

	# build model
	print("buidling model ...")
	model = args.model(args.config.model, dummy_dataset)

	# reload checkpoint
	print("restoring ckpt ...")
	restore_modules = {'model' : model}
	_, _ = restore(args.restore, 
					restore_modules, 
					num_checkpoints=args.average_checkpoints,
		            map_location=args.device.type,
		            strict=False
				)

	print("computing scores ...")
	scores = get_scores(model, data['batches'])


	with open(os.path.join(SCORES_PATH, score_fname), "w") as f:
		for s in scores:
			f.write(str(s) + "\n")


