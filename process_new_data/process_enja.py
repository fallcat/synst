"""
	


"""
import os
import sys
import array
import struct
import numpy as np
import sentencepiece as spm

# SAVE PATH
# RAW_DIR = "/mnt/nfs/work1/miyyer/simengsun/data/small_enja/"
# SAVE_DIR = "/mnt/nfs/work1/miyyer/simengsun/data/small_enja/"

SAVE_DIR = ""
RAW_DIR = ""

np.random.seed(42)


def save_data(fold, lang, data):
	with open(os.path.join(SAVE_DIR, "{}.tok.{}".format(fold, lang)), "w") as f:
		f.writelines(data)

def load_vocab():
	"""vocab, basically token2id"""
	vocab_path = os.path.join(SAVE_DIR, "vocab.bpe.32000")
	vocab = {}
	with open(vocab_path, "r") as f:
		for idx, line in enumerate(f.readlines()):
			token = line.strip()
			vocab[token] = idx
		vocab[''] = len(vocab)
	return vocab

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
	tgt_path = os.path.join(SAVE_DIR, f'{fold}.tok.bpe.32000.ja')
	src_path = os.path.join(SAVE_DIR, f'{fold}.tok.bpe.32000.en')
	out_path = os.path.join(SAVE_DIR, f'{fold}.tok.bpe.bin')

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

def spm_bpe_encode(sp, fold, lang):

	raw_path = os.path.join(RAW_DIR, f'{fold}.tok.{lang}')
	with open(raw_path, "r") as f:
		lines = f.readlines()
	out_path = os.path.join(SAVE_DIR, f'{fold}.tok.bpe.32000.{lang}')
	with open(out_path, "w") as f:
		for idx, line in enumerate(lines):
			f.write(' '.join(sp.EncodeAsPieces(line)) + "\n")
			if idx % 1000 == 0:
				print("end {} {} {}".format(fold, lang, idx))

def process_vocab(vocab_path):

	with open(vocab_path, 'r') as f:
		lines = f.readlines()

	out_vocab_path = os.path.join(SAVE_DIR, "vocab.bpe.32000")
	with open(out_vocab_path, 'w') as f:
		for line in lines[3:]:
			f.write(line.split()[0] + '\n')

if __name__ == "__main__":

	# Learn sentence piece model
	# >>> import sentencepiece as spm
	# >>> spm.SentencePieceTrainer.Train('--input=test/botchan.txt --model_prefix=m --vocab_size=1000')

	print(sys.argv)
	RAW_DIR = sys.argv[2]
	SAVE_DIR = sys.argv[3]

	if sys.argv[1] == "binarize":
		vocab = load_vocab()
		binarize("train", vocab)
		binarize("test", vocab)
		binarize("valid", vocab)

	elif sys.argv[1] == "spm_bpe_encode":
		sp = spm.SentencePieceProcessor()
		sp.Load(os.path.join(SAVE_DIR, 'spm.bpe.model'))
		spm_bpe_encode(sp, "train", "en")
		spm_bpe_encode(sp, "test", "en")
		spm_bpe_encode(sp, "valid", "en")
		spm_bpe_encode(sp, "train", "ja")
		spm_bpe_encode(sp, "test", "ja")
		spm_bpe_encode(sp, "valid", "ja")

	elif sys.argv[1] == "process_spm_vocab":
		process_vocab(os.path.join(SAVE_DIR, 'spm.bpe.vocab'))
