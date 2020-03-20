#!/bin/bash

# pjpcess ja corpus
# 1. replace space with special token, 2. tokenize (characters separated by space)
# python pjpcess_enjp.py pjpcess_jp

# pjpcess en corpus
# get bpe codes and vocab

# joint vocab 

# binarize data


# data path
BASE_PATH=/mnt/nfs/work1/miyyer/simengsun/data/small_enjp/
EN_TRAIN_RAW=$BASE_PATH/train.raw.en
JA_TRAIN_RAW=$BASE_PATH/train.raw.ja
EN_TRAIN_TOK=$BASE_PATH/train.tok.en
JA_TRAIN_TOK=$BASE_PATH/train.tok.ja
EN_DEV_RAW=$BASE_PATH/dev.raw.en
JA_DEV_RAW=$BASE_PATH/dev.raw.ja
EN_TEST_RAW=$BASE_PATH/test.raw.en
JA_TEST_RAW=$BASE_PATH/test.raw.ja
EN_DEV_TOK=$BASE_PATH/valid.tok.en
JA_DEV_TOK=$BASE_PATH/valid.tok.ja
EN_TEST_TOK=$BASE_PATH/test.tok.en
JA_TEST_TOK=$BASE_PATH/test.tok.ja

# sennrich's script for prepjpcess jpmanian
WMT16_SCRIPTS=/mnt/nfs/work1/miyyer/simengsun/other/wmt16-scripts
NORMALIZE_JAMANIAN=$WMT16_SCRIPTS/prepjpcess/normalise-jpmanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/prepjpcess/remove-diacritics.py
MOSES=/mnt/nfs/work1/miyyer/simengsun/other/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FJAM_SGM=$MOSES/scripts/ems/support/input-fjpm-sgm.perl
N_THREADS=8

EN_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $EN_RAW | $REM_NON_PRINT_CHAR | $TOKENIZER -l $EN_RAW -no-escape -threads $N_THREADS"
JA_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $JA_RAW | $REM_NON_PRINT_CHAR | $TOKENIZER -l $JA_RAW -no-escape -threads $N_THREADS"

cat $EN_TRAIN_RAW | $EN_PREPROCESSING > $EN_TRAIN_TOK
cat $EN_DEV_RAW | $EN_PREPROCESSING > $EN_DEV_TOK
cat $EN_TEST_RAW | $EN_PREPROCESSING > $EN_TEST_TOK
cat $JA_TRAIN_RAW | $JA_PREPROCESSING > $JA_TRAIN_TOK
cat $JA_DEV_RAW | $JA_PREPROCESSING > $JA_DEV_TOK
cat $JA_TEST_RAW | $JA_PREPROCESSING > $JA_TEST_TOK

# Learn sentence piece model
# >>> import sentencepiece as spm
# >>> spm.SentencePieceTrainer.Train('--input=test/botchan.txt --model_prefix=m --vocab_size=1000')