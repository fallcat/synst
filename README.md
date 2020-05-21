# StupidNMT: Hard-Coded Gaussian Attention for Neural Machine Translation

This is the official repository which contains all the code necessary to
replicate the results from the ACL 2020 long paper *[Hard-Coded Gaussian Attention for Neural Machine Translation](https://arxiv.org/abs/2005.00742)*. It can also be used to
train a vanilla Transformer.

The full model architecture is displayed below:

<p>
<img src="resources/model.pdf">
</p>

![image](resources/model.png)

Our approach uses hard-coded Gaussian distribution instead of learned attention to simplify the Transformer architecture in neural machine translation (NMT). We replace the multi-headed attention, computed by query and key, by a fixed Gaussian distribution that focuses on the current word or somewhere near it. The figure above demonstrates how our attention differs from the vanilla Tranformer.

This code base is adapted from [synst](https://github.com/dojoteef/synst).

## Requirements

The code requires Python 3.7+. The python dependencies can be installed with the
command (using a virtual environment is highly recommended):

```sh
pip install -r requirements.txt
```

If you want to use the scripts that wrap `multi-bleu.perl` and
`sacrebleu`, then you'll need to have
[Moses-SMT](https://github.com/moses-smt/mosesdecoder) available as well.

## Basic Usage

The code has one main entry point `main.py` with a couple of support scripts for
the analysis conducted in the paper. Please use `python main.py -h` for
additional options not listed below. You can also use `python main.py <action>
-h` for options specific to the available actions: `{train, evaluate, translate,
pass}`.

### Preprocessing

```sh
CLASSPATH=stanford-corenlp-full-2018-10-05/* python main.py \
  --dataset wmt_en_de_parsed --span 6 -d raw/wmt -p preprocessed/wmt -v pass
```

### Training

Assuming you have access to 8 1080Ti GPUs you can recreate the results for stupidNMT
on the WMT'14 En-De dataset with:

```sh
python main.py -b 3175 --dataset wmt_en_de_parsed --span 6 \
  --model new_transformer -d raw/wmt -p preprocessed/wmt -v train \
  --checkpoint-interval 1200 --accumulate 2 --label-smoothing 0
```

The above commandline will train 8 GPUs with approximately 3175 source/target
tokens combined per GPU, and accumulate the gradients over two batches before
updating model parameters (leading to ~50.8k tokens per model update).

The default model is the Transformer model. For example the below
line will train a vanilla Transformer on the WMT'14 De-En
dataset:

```sh
python main.py -b 3175 --dataset wmt_de_en \
  -d raw/wmt -p preprocessed/wmt -v train \
  --checkpoint-interval 1200 --accumulate 2
```

To train a hard-coded self-attention model, you can run this:

```sh
python main.py -b 3175 --dataset wmt_de_en \
  --model new_transformer \
  --enc-attn-type normal --enc-attn-offset -1 1 \
  --dec-attn-type normal --dec-attn-offset -1 0 \
  -d raw/wmt -p preprocessed/wmt -v train \
  --checkpoint-interval 1200 --accumulate 2
```

### Evalulating Perplexity

You can run a separate process to evaluate each new checkpoint generated during
training (you may either want to do it on a GPU not used for training or disable
cuda as done below):

```sh
python main.py -b 5000 --dataset wmt_en_de \
  --model new_transformer -d raw/wmt -p preprocessed/wmt \
  --enc-attn-type normal --enc-attn-offset -1 1 \
  --dec-attn-type normal --dec-attn-offset -1 0 \
  --split valid --disable-cuda -v evaluate \
  --watch-directory /tmp/stupidnmt/checkpoints
```

### Translating

After training a model, you can generate translations with the following
command (currently only translation on a single GPU is supported):

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --dataset wmt_en_de \
  --model new_transformer \
  --enc-attn-type normal --enc-attn-offset -1 1 \
  --dec-attn-type normal --dec-attn-offset -1 0 \
  -d raw/wmt -p preprocessed/wmt \
  --batch-size 1 --batch-method example --split test -v \
  --restore /tmp/stupidnmt/checkpoints/checkpoint.pt \
  --average-checkpoints 5 translate \
  --max-decode-length 50 --length-basis input_lens --order-output
```

Which by default, will output translations to `/tmp/stupidnmt/output`.

### Experiment tracking

If you have a [comet.ml](https://comet.ml) account, on you can track
experiments, by prefixing the script call with:

```sh
env $(cat ~/.comet.ml | xargs) python main.py --track ...
```

Where `~/.comet.ml` is the file which contains your API key for logging
experiments on the service. By default, this will track experiments in a
workspace named `umass-nlp` with project name `probe-transformer`. See `args.py` in order to
configure the experiment tracking to suit your needs.

## Cite (Will be updated once the ACL Anthology version comes out)

```bibtex
@misc{you2020hardcoded,
    title={Hard-Coded Gaussian Attention for Neural Machine Translation},
    author={Weiqiu You and Simeng Sun and Mohit Iyyer},
    year={2020},
    eprint={2005.00742},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
