#!/bin/bash

# # sample small size data
# python process_enfr.py sample

# BPE related 
SUBWORD=/mnt/nfs/work1/miyyer/simengsun/other/subword-nmt/subword_nmt/
BPE_VOCAB=/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/vocab.bpe.freq.32000
BPE_CODES=/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/bpe.32000
MERGE_OPS=32000

# learn shared bpe codes from small en-fr
BASE_PATH=/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/
EN_TOK=/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/train.tok.en
FR_TOK=/mnt/nfs/work1/miyyer/simengsun/data/small_enfr/train.tok.fr

# cat $EN_TOK $FR_TOK | $SUBWORD/learn_bpe.py -s $MERGE_OPS > $BPE_CODES

# get vocab

# vocab with word frequency
cat $BASE_PATH/train.tok.bpe.$MERGE_OPS.en $BASE_PATH/train.tok.bpe.$MERGE_OPS.fr \
	$BASE_PATH/test.tok.bpe.$MERGE_OPS.en $BASE_PATH/test.tok.bpe.$MERGE_OPS.fr \
	$BASE_PATH/valid.tok.bpe.$MERGE_OPS.en $BASE_PATH/valid.tok.bpe.$MERGE_OPS.fr \
	> $BASE_PATH/joint.tok.bpe.$MERGE_OPS
$SUBWORD/get_vocab.py --input $BASE_PATH/joint.tok.bpe.$MERGE_OPS --output $BASE_PATH/vocab.bpe.freq.$MERGE_OPS

# w/o freq
cat $BASE_PATH/vocab.bpe.freq.$MERGE_OPS | cut -f 1 -d ' ' > $BASE_PATH/vocab.bpe.$MERGE_OPS

# encode text using the bpe codes, add vocab for reverse encoding
for lang in en fr; do
	for f in $BASE_PATH/*.tok.$lang; do
		outfile=${f%.*}.bpe.$MERGE_OPS.$lang
		$SUBWORD/apply_bpe.py -c $BPE_CODES --vocabulary $BPE_VOCAB < $f > $outfile
		echo $outfile
	done
done

# binarize bpe encoded data
python process_enfr.py binarize