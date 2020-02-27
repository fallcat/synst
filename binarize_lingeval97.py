

"""
	
	Get binarized data of lingeval97.bpe.32000 
	Store to pickle so evaluator can read directly
	Each batch object is a dictionary:

	{'example_ids': (0,), 
	'inputs': tensor([[ 9846,  3826, 39, 24238,  8223,  1308, 14, 30483,   962]]), 
	'input_lens': tensor([9]), 
	'targets': tensor([[37006,  9846, 3826, 39,  8545, 189, 846, 19, 30928,  1045, 37007]]), 
	'target_lens': tensor([11])}

	bc in evaluate mode, each batch contains ONE example

"""

import torch
import pickle
from datetime import datetime

PAD = '<PAD>'
BOS = '<SOS>'
EOS = '<EOS>'

def load_vocab():
	'''load vocab'''
	vocab_path = "/mnt/nfs/work1/miyyer/wyou/wmt/vocab.bpe.32000"
	token2id, id2token = {}, []
	with open(vocab_path, 'r') as f:
		for line in f.readlines():
			token = line.strip()
			token2id[token] = len(id2token)
			id2token.append(token)
	token2id[''] = len(id2token)
	id2token.append('')
	token2id[PAD] = len(id2token)
	id2token.append(PAD)
	token2id[BOS] = len(id2token)
	id2token.append(BOS)
	token2id[EOS] = len(id2token)
	id2token.append(EOS)
	return token2id, id2token

def tensorize(sent, token2id, tgt=False):
	'''turn each token to idx given token2id, if tgt, add EOS and BOS'''
	if tgt:
		ret = [token2id[BOS]]
	else:
		ret = []
	for tok in sent.strip().split():
		ret.append(token2id[tok])
	if tgt:
		ret.append(token2id[EOS])
	return torch.LongTensor([ret])

def binarize():

	# load vocab
	print("loading vocab ...")
	token2id, id2token = load_vocab()

	# load src, tgt sentences
	print("processing data ...")
	src_bpe_path = "/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.en.synst.tok.32000.bpe"
	tgt_bpe_path = "/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.de.synst.tok.32000.bpe"

	batches = []
	with open(src_bpe_path, 'r') as f_src, open(tgt_bpe_path, 'r') as f_tgt:
		src_lines, tgt_lines = f_src.readlines(), f_tgt.readlines()
		assert len(src_lines) == len(tgt_lines), print("src len {} tgt len {}".format(len(src_lines), len(tgt_lines)))
		for idx, (s_sent, t_sent) in enumerate(zip(src_lines, tgt_lines)):
			s_tensor = tensorize(s_sent, token2id)
			t_tensor = tensorize(t_sent, token2id, tgt=True)
			s_lens = torch.LongTensor([s_tensor.shape[1]])
			t_lens = torch.LongTensor([t_tensor.shape[1]])
			it = {}
			it['example_ids'] = (idx,)
			it['inputs'] = s_tensor
			it['targets'] = t_tensor
			it['input_lens'] = s_lens
			it['target_lens'] = t_lens
			batches.append(it)
			if idx % 500 == 0:
				print("{} {}".format(datetime.now(), idx))

	# store as pickle
	lingeval97_bin_path = "/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.synst.32000.pkl"
	with open(lingeval97_bin_path, 'wb') as f:
		pickle.dump({'batches': batches}, f)


if __name__ == "__main__":

	binarize()
