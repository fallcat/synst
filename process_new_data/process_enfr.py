"""

	Randomly sample from wmt_enfr data the size of iwslt en-de data

	Specifically, 
		    1. Train: 196884
		    2. Valid: 7883
		    3. Test: 2762

"""
import os
import sys
import array
import struct
import numpy as np

# BASE DIR
BASE_PATH = "/mnt/nfs/work1/miyyer/nsa/faster-decoder/data/wmt_enfr/"
# SAVE PATH
SAVE_PATH = "/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/"

N_TRAIN = 196884
N_VAL = 7883
N_TEST = 2762
np.random.seed(42)

def load_data(fold, lang):
	with open(os.path.join(BASE_PATH, "{}.tok.{}".format(fold, lang)), "r") as f:
		lines = f.readlines()
	return lines

def save_data(fold, lang, data):
	with open(os.path.join(SAVE_PATH, "{}.tok.{}".format(fold, lang)), "w") as f:
		f.writelines(data)

def load_vocab():
	"""vocab, basically token2id"""
	vocab_path = "/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/vocab.bpe.32000"
	vocab = {}
	with open(vocab_path, "r") as f:
		for idx, line in enumerate(f.readlines()):
			token = line.strip()
			vocab[token] = idx
		vocab[''] = len(vocab)
	return vocab

def sample_and_save(fold, num):
	"""random sample num sentences from fold {train/test/valid}"""
	print(f"loading {fold} data ...")
	data_en, data_fr = load_data(fold, "en"), load_data(fold, "fr")
	assert len(data_en) == len(data_fr)
	try:
		sample_idx = np.random.choice(len(data_en), num, replace=False)
	except:
		sample_idx =np.random.choice(len(data_en), len(data_en), replace=False)
	data_en_sample = [data_en[i] for i in sample_idx]
	data_fr_sample = [data_fr[i] for i in sample_idx]
	print("saving en data ...")
	save_data(fold, "en", data_en_sample)
	print("saving fr data ...")
	save_data(fold, "fr", data_fr_sample)

def tensorize(sent, vocab):

	sent_enc = array.array('H')
	sent_enc.extend((vocab[token] for token in sent.split()))
	byte_rep = sent_enc.tobytes()
	byte_len = len(byte_rep)
	return struct.pack('Q{}s'.format(byte_len), byte_len, byte_rep)

def binarize(fold, vocab):
	"""
		
		binarize the bpe-ed file to .bin which will be read by the model
		preprocess steps: synst.src.data.annotated.preprocess_bpe

	"""
	tgt_path = os.path.join(SAVE_PATH, f'{fold}.tok.bpe.32000.fr')
	src_path = os.path.join(SAVE_PATH, f'{fold}.tok.bpe.32000.en')
	out_path = os.path.join(SAVE_PATH, f'{fold}.tok.bpe.bin')

	with open(src_path, 'r') as f_src, \
			open(tgt_path, 'r') as f_tgt, \
			open(out_path, 'wb') as f_out:
		src_data = f_src.readlines()
		tgt_data = f_tgt.readlines()
		assert len(src_data) == len(tgt_data)

		for sent_s, sent_t in zip(src_data, tgt_data):
			sent_s = tensorize(sent_s, vocab)
			sent_t = tensorize(sent_t, vocab)
			f_out.write(sent_s)
			f_out.write(sent_t)

if __name__ == "__main__":

	if sys.argv[1] == "sample":
		sample_and_save("train", N_TRAIN)
		sample_and_save("test", N_TEST)
		sample_and_save("valid", N_VAL)

	elif sys.argv[1] == "binarize":
		vocab = load_vocab()
		binarize("train", vocab)
		binarize("test", vocab)
		binarize("valid", vocab)


