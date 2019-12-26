#!/bin/bash

# Learn bpe with codes=32000 from wmt en-de data
# apply bpe codes to lingeval97 data




SRC_TOK=/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.synst.tok.en
TGT_TOK=/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.synst.tok.de
RAW_BASE=/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97

# synst version of tokenizer
MOSES=/mnt/nfs/work1/miyyer/simengsun/xlm/XLM/tools/mosesdecoder
$MOSES/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < ${RAW_BASE}.en > ${RAW_BASE}.synst.tok.en
$MOSES/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < ${RAW_BASE}.de > ${RAW_BASE}.synst.tok.de

OUTPUT_DIR=/mnt/nfs/work1/miyyer/simengsun/other/
# Clone Subword NMT
if [ ! -d ${OUTPUT_DIR}/subword-nmt ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# BPE codes
BPE_CODES=/mnt/nfs/work1/miyyer/simengsun/data/wmt_ende/bpe.32000
BPE_VOCAB=/mnt/nfs/work1/miyyer/simengsun/data/wmt_ende/vocab.bpe.dummy.32000

SRC_BPE=/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.en.synst.tok.32000.bpe
TGT_BPE=/mnt/nfs/work1/miyyer/simengsun/lingeval97/lingeval97.de.synst.tok.32000.bpe

# apply BPE codes
python ${OUTPUT_DIR}/subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODES --vocabulary $BPE_VOCAB --vocabulary-threshold 5 < $SRC_TOK > $SRC_BPE
python ${OUTPUT_DIR}/subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODES --vocabulary $BPE_VOCAB --vocabulary-threshold 5 < $TGT_TOK > $TGT_BPE
