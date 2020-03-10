'''
All the argument parsing code
'''
import argparse
import json
import os
import time
from types import SimpleNamespace

from comet_ml import Experiment, ExistingExperiment
import torch

from data import DATASETS
from models import MODELS
from actions import Trainer, Evaluator, Translator, Pass, Prober, ProbeTrainer, ProbeEvaluator, ProbeNewTranslator, \
    ProbeOffDiagonal, IterativeTrainer
from utils import get_version_string, get_random_seed_fn


def _integer_geq(value=0):
    '''
    Return a function which when evaluated returns an integer if a passed in string when converted
    to an int is greater than or equal to the specified constant. Otherwise it raises an error.
    '''
    def convert(string):
        ''' Check if the string is an integer that is greater than value '''
        integer = int(string)
        if integer < value:
            raise argparse.ArgumentTypeError(f'{string} should be >= {value}')
        return integer

    return convert


class ArgGroup(object):
    ''' A simple class wrapping argparse groups '''
    def __init__(self, group):
        self.args = []
        self.dict = {}
        self.group = group

    def __getattr__(self, name):
        ''' Try to read an argument '''
        if name == 'dict':
            return self.dict
        else:
            return self.dict.get(name)

    def __str__(self):
        ''' Return a json representation of the arguments '''
        return json.dumps(self.dict, indent=1)

    def set_defaults(self, **kwargs):
        ''' Set the defaults based on the kwargs '''
        self.dict.update(kwargs)

    def read(self, args):
        ''' Read in the parsed args that are part of this group '''
        self.dict.update({k: args.__dict__[k] for k in self.args if k in args.__dict__})

        return self

    def add_argument(self, *args, **kwargs):
        ''' Add an argument to group '''
        # pylint:disable=protected-access
        actions = set(self.group._actions)
        self.group.add_argument(*args, **kwargs)
        for action in set(self.group._actions) - actions:
            self.args.append(action.dest)
        # pylint:enable=protected-access


def add_transformer_args(parser):
    ''' Defines Transformer model specific arguments '''
    group = ArgGroup(parser.add_argument_group('Transformer Model'))
    group.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of layers in each Transformer stack'
    )
    group.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of heads in each Transformer layer for multi-headed attention'
    )
    group.add_argument(
        '--embedding-size',
        type=int,
        default=512,
        help='The size of the Transformer model dimension'
    )
    group.add_argument(
        '--hidden-dim',
        type=int,
        default=2048,
        help='The size of the Transformer feed-forward hidden layer'
    )
    group.add_argument(
        '--span',
        type=int,
        default=1,
        help='How many tokens to decode at once'
    )

    return group


def add_probe_transformer_args(parser):
    ''' Defines Transformer model specific arguments '''
    group = ArgGroup(parser.add_argument_group('Probe Transformer Model'))
    group.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of layers in each Transformer stack'
    )
    group.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of heads in each Transformer layer for multi-headed attention'
    )
    group.add_argument(
        '--embedding-size',
        type=int,
        default=512,
        help='The size of the Transformer model dimension'
    )
    group.add_argument(
        '--hidden-dim',
        type=int,
        default=2048,
        help='The size of the Transformer feed-forward hidden layer'
    )
    group.add_argument(
        '--span',
        type=int,
        default=1,
        help='How many tokens to decode at once'
    )

    return group


def add_new_transformer_args(parser):
    ''' Defines Transformer model specific arguments '''
    group = ArgGroup(parser.add_argument_group('New Transformer Model'))
    group.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of layers in each Transformer stack'
    )
    group.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of heads in each Transformer layer for multi-headed attention'
    )
    group.add_argument(
        '--embedding-size',
        type=int,
        default=512,
        help='The size of the Transformer model dimension'
    )
    group.add_argument(
        '--hidden-dim',
        type=int,
        default=2048,
        help='The size of the Transformer feed-forward hidden layer'
    )
    group.add_argument(
        '--span',
        type=int,
        default=1,
        help='How many tokens to decode at once'
    )
    group.add_argument(
        '--attn-type',
        type=str,
        nargs='+',
        default='normal',
        choices=['normal', 'uniform', 'no', 'learned'],
        help='What type of attention we are using for the rules'
    )
    group.add_argument(
        '--attn-position',
        type=str,
        nargs='+',
        default='center',
        choices=['center', 'left', 'right', 'first', 'last', 'bin'],
        help='Where to put the attention. Center is centered at the word.'
    )
    group.add_argument(
        '--attn-param',
        type=float,
        nargs='+',
        default=1,
        help='when attention type is normal. The standard deviation.'
             'when attention type is uniform. The number of tokens to focus on to each direction.'
             'If window size is 2, then we have uniform distribution on 5 words.'
    )
    group.add_argument(
        '--attn-threshold',
        type=float,
        default=-1,
        help='when attention type is normal, '
             'Only attend to places where distribution is above or equal to the attn-threshold'
    )
    group.add_argument(
        '--attn-window',
        type=int,
        default=-1,
        help='The window to do convolution at'
    )
    group.add_argument(
        '--attn-displacement',
        type=int,
        nargs='+',
        default=1,
        help='Only works when corresponding attn-position is left or right.'
             'The number of steps to take to that direction.'
             'When attn-position is bin, this number has to be between 1 and number of attn-bins'
    )
    group.add_argument(
        '--attn-concat',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='Whether or not concat previous embedding with new embedded. 0 to not concat, '
             '1 to concat just previous embedding, 2 to concat just word embedding, 3 to concat both.'
    )
    group.add_argument(
        '--attn-weights',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Whether or not use weights in non-learned attention. 0 no weights, 1 all with weights, '
             '2 weight only for key and query but not value.'
    )
    group.add_argument(
        '--attn-score',
        type=int,
        default=0,
        choices=[0, 1],
        help='Whether or not use query to score the different heads. 0 do not score, 1 do score.'
    )
    group.add_argument(
        '--attn-bins',
        type=int,
        default=2,
        help='Number of bins to look at in total'
    )

    group.add_argument(
        '--dec-attn-type',
        type=str,
        nargs='+',
        default='learned',
        choices=['normal', 'uniform', 'no', 'learned'],
        help='What type of attention we are using for the rules'
    )
    group.add_argument(
        '--dec-attn-position',
        type=str,
        nargs='+',
        default='center',
        choices=['center', 'left', 'right', 'first', 'last', 'bin'],
        help='Where to put the attention. Center is centered at the word.'
    )
    group.add_argument(
        '--dec-attn-param',
        type=float,
        nargs='+',
        default=1,
        help='when attention type is normal. The standard deviation.'
             'when attention type is uniform. The number of tokens to focus on to each direction.'
             'If window size is 2, then we have uniform distribution on 5 words.'
    )
    group.add_argument(
        '--dec-attn-threshold',
        type=float,
        default=-1,
        help='when attention type is normal, '
             'Only attend to places where distribution is above or equal to the attn-threshold'
    )
    group.add_argument(
        '--dec-attn-window',
        type=int,
        default=-1,
        help='The window to do convolution at'
    )
    group.add_argument(
        '--dec-attn-displacement',
        type=int,
        nargs='+',
        default=1,
        help='Only works when corresponding attn-position is left or right.'
             'The number of steps to take to that direction.'
             'When attn-position is bin, this number has to be between 1 and number of attn-bins'
    )
    group.add_argument(
        '--dec-attn-concat',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='Whether or not concat previous embedding with new embedded. 0 to not concat, '
             '1 to concat just previous embedding, 2 to concat just word embedding, 3 to concat both.'
    )
    group.add_argument(
        '--dec-attn-weights',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Whether or not use weights in non-learned attention. 0 no weights, 1 all with weights, '
             '2 weight only for key and query but not value.'
    )
    group.add_argument(
        '--dec-attn-score',
        type=int,
        default=0,
        choices=[0, 1],
        help='Whether or not use query to score the different heads. 0 do not score, 1 do score.'
    )
    group.add_argument(
        '--dec-attn-bins',
        type=int,
        default=2,
        help='Number of bins to look at in total'
    )

    group.add_argument(
        '--enc-dec-attn-type',
        type=str,
        nargs='+',
        default='learned',
        choices=['normal', 'uniform', 'no', 'learned'],
        help='What type of attention we are using for the rules'
    )
    group.add_argument(
        '--enc-dec-attn-position',
        type=str,
        nargs='+',
        default='center',
        choices=['center', 'left', 'right', 'first', 'last', 'bin'],
        help='Where to put the attention. Center is centered at the word.'
    )
    group.add_argument(
        '--enc-dec-attn-param',
        type=float,
        nargs='+',
        default=1,
        help='when attention type is normal. The standard deviation.'
             'when attention type is uniform. The number of tokens to focus on to each direction.'
             'If window size is 2, then we have uniform distribution on 5 words.'
    )
    group.add_argument(
        '--enc-dec-attn-threshold',
        type=float,
        default=-1,
        help='when attention type is normal, '
             'Only attend to places where distribution is above or equal to the attn-threshold'
    )
    group.add_argument(
        '--enc-dec-attn-window',
        type=int,
        default=-1,
        help='The window to do convolution at'
    )
    group.add_argument(
        '--enc-dec-attn-displacement',
        type=int,
        nargs='+',
        default=1,
        help='Only works when corresponding attn-position is left or right.'
             'The number of steps to take to that direction.'
             'When attn-position is bin, this number has to be between 1 and number of attn-bins'
    )

    group.add_argument(
        '--enc-dec-attn-concat',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='Whether or not concat previous embedding with new embedded. 0 to not concat, '
             '1 to concat just previous embedding, 2 to concat just word embedding, 3 to concat both.'
    )

    group.add_argument(
        '--enc-dec-attn-weights',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Whether or not use weights in non-learned attention. 0 no weights, 1 all with weights, '
             '2 weight only for key and query but not value.'
    )

    group.add_argument(
        '--enc-dec-attn-score',
        type=int,
        default=0,
        choices=[0, 1],
        help='Whether or not use query to score the different heads. 0 do not score, 1 do score.'
    )
    group.add_argument(
        '--enc-dec-attn-bins',
        type=int,
        default=2,
        help='Number of bins to look at in total'
    )

    group.add_argument(
        '--enc-dec-attn-layer',
        type=int,
        nargs='+',
        default=1,
        choices=[0, 1],
        help="Determine the presence of encoder-decoder (source) attention after the decoder self-attention layer. "
             "Length of list should be equal to the number of layers. "
    )

    group.add_argument(
        '--enc-dec-attn-num-heads',
        type=int,
        nargs='+',
        default=-1,
        help="number of heads for enc-dec-attn at each layer; length of list should be equal to the number of layers. "
             "if a layer does not have source attention, it should be 0."
             "If the number of heads is same as encoder/decoder self attentions, then use -1 to specify that."
    )

    group.add_argument(
        '--enc-attn-indexing',
        type=bool,
        default=False,
        help="flag indicating whether use indexing(window-size=1) or not. if use, center+displacement word will be selected directly from values."
    )

    group.add_argument(
        '--dec-attn-indexing',
        type=bool,
        default=False,
        help="flag indicating whether use indexing(window-size=1) or not. if use, center+displacement word will be selected directly from values."
    )

    group.add_argument(
        '--enc-dec-attn-indexing',
        type=bool,
        default=False,
        help="flag indicating whether use indexing(window-size=1) or not. if use, center+displacement word will be selected directly from values."
    )

    group.add_argument(
        '--indexing-type',
        type=str,
        choices=['gather', 'bmm'],
        default='bmm'
    )

    group.add_argument(
        '--enc-no-attn',
        default=False,
        action='store_true',
        help="flag indicating not using attention for encoder self-attention"
    )

    group.add_argument(
        '--dec-no-attn',
        default=False,
        action='store_true',
        help="flag indicating not using attention for decoder self-attention"
    )

    group.add_argument(
        '--tie-ffn-weights',
        type=bool,
        help="whether tie ffn layer weights or not",
        default=False
    )

    group.add_argument(
        '--ffn-layer',
        type=int,
        nargs='+',
        help="which layer has ffn layer, 0 no 1 yes",
        default=-1
    )

    #group.add_argument(
     #   '--layer-mask',
      #  default=False,
       # action='store_true',
       # help="Use layer mask"
    #)

    group.add_argument(
        '--gating-tradeoff',
        type=float,
        default=2.0,
        help='gating tradeoff'
    )
    
    group.add_argument(
        '--random-layermask',
        default=False,
        action='store_true',
        help="whether to use random layermask, if true, random sample layermask, not updating layermask predictor"
    )

    group.add_argument(
        '--diversity-tradeoff',
        type=float,
        default=0.2,
        help="for diversity regularizer"
    )

    group.add_argument(
        '--layermask-type',
        type=str,
        choices=['noskip', 'gating', 'iterative_training','iterative_training_debug_oracle'],
        default='noskip'
    )
    group.add_argument(
        '--layermask-noisy',
        default=False,
        action='store_true',
        help="whether to add noisy during generating layermasks"
    )
    group.add_argument(
        '--allon-threshold',
        type=float,
        default=0.3,
        help="threshold smaller than which will switch to all-on config"
    )

    group.add_argument(
        '--potential-threshold',
        type=float,
        default=0.1,
        help="threshold smaller than which will switch to all-on config"
    )

    group.add_argument(
        '--shuffle-lmp-configs',
        default=False,
        action="store_true",
        help="whether to shuffle the configs, used for optimizing the selected config range"
    )

    group.add_argument(
        "--num-configs",
        default=4032,
        type=int,
        help="number of configs to optimize"
    )

    return group


def add_parse_transformer_args(parser):
    ''' Defines Transformer model specific arguments '''
    group = ArgGroup(parser.add_argument_group('Transformer Model'))
    group.add_argument(
        '--parse-only',
        default=False,
        action='store_true',
        help='Whether to only model the parse'
    )
    group.add_argument(
        '--parse-num-layers',
        type=int,
        default=1,
        help='Number of layers in the Parse decoder stack'
    )
    group.add_argument(
        '--num-layers',
        type=int,
        default=6,
        help='Number of layers in each Transformer stack'
    )
    group.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of heads in each Transformer layer for multi-headed attention'
    )
    group.add_argument(
        '--embedding-size',
        type=int,
        default=512,
        help='The size of the Transformer model dimension'
    )
    group.add_argument(
        '--hidden-dim',
        type=int,
        default=2048,
        help='The size of the Transformer feed-forward hidden layer'
    )
    group.add_argument(
        '--span',
        type=int,
        default=1,
        help='How many tokens to decode at once'
    )

    return group


def add_data_args(parser):
    ''' Defines the preprocessing specific arguments '''
    group = ArgGroup(parser.add_argument_group('Data and Preprocessing'))
    group.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=8000,
        help='Maximum number of tokens per batch'
    )
    group.add_argument(
        '--batch-size-buffer',
        type=int,
        default=0,
        help='By how many tokens to reduce the batch size on the GPU of the optimizer'
    )
    group.add_argument(
        '--batch-method',
        type=str,
        default='token',
        choices=['token', 'example'],
        help='Whether to batch by number of tokens or examples'
    )
    group.add_argument(
        '-g',
        '--token-bucket-granularity',
        type=int,
        default=3,
        help='Granularity of each bucket for the token based batching method'
    )
    group.add_argument(
        '-d',
        '--data-directory',
        type=str,
        default='/tmp/synst/data',
        help='Location of the data'
    )
    group.add_argument(
        '-D',
        '--dataset',
        type=str,
        default='wmt_en_de',
        choices=DATASETS,
        help='Name of the dataset to load. Defaults to wmt_en_de'
    )
    group.add_argument(
        '--max-line-length',
        type=int,
        default=0,
        help='Maximum line length during data preprocessing. Throw out lines greater than this.'
    )
    group.add_argument(
        '--max-input-length',
        type=int,
        default=0,
        help='Maximum input tokens per example'
    )
    group.add_argument(
        '--max-target-length',
        type=int,
        default=0,
        help='Maximum target tokens per example'
    )
    group.add_argument(
        '--max-examples',
        type=int,
        default=0,
        help='Maximum number of training examples. Defaults to all of them'
    )
    group.add_argument(
        '--max-span',
        type=int,
        default=0,
        help='Any training example with span larger than the max span is skipped.'
    )
    group.add_argument(
        '--preprocess-buffer-size',
        type=int,
        default=12500,
        help='Number of lines to preprocess at once'
    )
    group.add_argument(
        '-p',
        '--preprocess-directory',
        type=str,
        default='/tmp/synst/data',
        help='Location for the preprocessed data'
    )
    group.add_argument(
        '--randomize-chunks',
        default=False,
        action='store_true',
        help='Whether to randomize the chunk size during training'
    )
    group.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'valid', 'test', 'dev','valid-train', 'valid-val'],
        help='Location for the preprocessed data'
    )
    group.add_argument(
        '--align-stats-bin-size',
        type=int,
        default=4,
        help='Bin size for word align stats'
    )

    group.add_argument(
        '--loss_func',
        type=str,
        default="binary_cls",
        help="objective for optimizing layermask-predictor"
    )

    group.add_argument(
        '--itertrain-data',
        type=str,
        default="none",
        #default="/mnt/nfs/work1/miyyer/simengsun/data/small_enro/valid.iter.50",
        help="location for the itertrain data"
    )

    return group

def add_cuda_args(parser):
    ''' Defines CUDA specific arguments '''
    group = ArgGroup(parser.add_argument_group('CUDA options'))
    group.add_argument(
        '--disable-cuda',
        default=False,
        action='store_true',
        help='Whether to disable CUDA. Use CUDA_VISIBLE_DEVICES= to limit available GPUs'
    )
    group.add_argument(
        '--profile-cuda',
        default=False,
        action='store_true',
        help='Whether to profile CUDA. Should be used in conjunction with: '
        "'nvprof --profile-from-start off -o trace_name.prof'"
    )
    group.add_argument(
        '--profile-cuda-memory',
        default=False,
        const='cuda.prof',
        nargs='?',
        type=str,
        help='Whether to profile CUDA memory.'
    )

    return group


def add_train_args(parser):
    ''' Defines the training specific arguments '''
    group = ArgGroup(parser.add_argument_group('Training'))
    group.add_argument(
        '-A',
        '--accumulate-steps',
        type=int,
        default=1,
        help='How many batches of data to accumulate gradients over'
    )
    group.add_argument(
        '--gold-p',
        type=float,
        default=1.0,
        help='The percentage of time to select gold targets during training (for LSTMs)'
    )
    group.add_argument(
        '--dropout-p',
        type=float,
        default=0.1,
        help='The dropout percentage during training'
    )
    group.add_argument(
        '--early-stopping',
        type=_integer_geq(),
        default=0,
        help='If > 0, stop training after this many checkpoints of increasing nll on the validation'
        ' set. This also implies storing of the best_checkpoint.'
    )
    group.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='The amount of label smoothing'
    )
    group.add_argument(
        '-c',
        '--checkpoint-directory',
        type=str,
        default='/tmp/synst/checkpoints',
        help='Where to store model checkpoints'
    )
    group.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10*60,
        help='Generate a checkpoint every `n` seconds'
    )
    group.add_argument(
        '--max-checkpoints',
        type=int,
        default=5,
        help='The maximum number of checkpoints to keep'
    )
    group.add_argument(
        '-e',
        '--max-epochs',
        type=int,
        default=0,
        help='Maximum number of epochs for training the model'
    )
    group.add_argument(
        '--max-steps',
        type=int,
        default=100000,
        help='Maximum number of steps for training the model'
    )
    group.add_argument(
        '-l',
        '--learning-rate',
        dest='base_lr',
        type=float,
        default=None,
        help='The initial learning rate of the optimizer. Defaults to embedding_size ** -0.5'
    )
    group.add_argument(
        '-L',
        '--learning-rate-decay',
        dest='lr_decay',
        type=float,
        default=.999995,
        help='The learning rate decay of the optimizer'
    )
    group.add_argument(
        '--final-learning-rate',
        dest='final_lr',
        type=float,
        default=1e-5,
        help='For the linear annealing schedule'
    )
    group.add_argument(
        '--learning-rate-scheduler',
        dest='lr_scheduler',
        type=str,
        default='warmup',
        choices=['exponential', 'warmup', 'linear','warmup2'],
        help='The learning rate schedule of the optimizer'
    )
    group.add_argument(
        '-w',
        '--warmup-steps',
        type=int,
        default=4000,
        help='Number of warmup steps for the Transformer learning rate'
    )
    group.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'adam-fixed'],
        help='add optimizer'
    )
    group.add_argument(
        '--freeze-layermask',
        default=False,
        action='store_true',
        help="whether to freeze the layermask predictor, this is set to True after training entire model to converge"
    )
    group.add_argument(
        '--linear-tradeoff',
        default=False,
        action='store_true',
        help="use linear scheduled tradeoffs"
    )
    

    return group


def add_iterative_train_args(parser):
    ''' Defines the training specific arguments '''
    group = ArgGroup(parser.add_argument_group('Training'))
    # part of original translate args
    group.add_argument(
        '--beam-width',
        default=4,
        type=int,
        help='Default beam width for beam search decoder.'
    )
    group.add_argument(
        '--length-penalty',
        type=float,
        default=0.6,
        help='Divides the hypothesis log probabilities in beam search by length^<length penalty>.'
    )
    group.add_argument(
        '--length-basis',
        type=str,
        default='input_lens',
        choices=['input_lens', 'target_lens'],
        help='The basis for max decoding length. Default of None implies no basis, i.e. 0.'
    )
    group.add_argument(
        '--max-decode-length',
        default=50,
        type=int,
        help='How many tokens beyond the length basis to allow decoding to continue.'
    )

    # original train args
    group.add_argument(
        '-A',
        '--accumulate-steps',
        type=int,
        default=1,
        help='How many batches of data to accumulate gradients over'
    )
    group.add_argument(
        '--gold-p',
        type=float,
        default=1.0,
        help='The percentage of time to select gold targets during training (for LSTMs)'
    )
    group.add_argument(
        '--dropout-p',
        type=float,
        default=0.1,
        help='The dropout percentage during training'
    )
    group.add_argument(
        '--early-stopping',
        type=_integer_geq(),
        default=0,
        help='If > 0, stop training after this many checkpoints of increasing nll on the validation'
        ' set. This also implies storing of the best_checkpoint.'
    )
    group.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='The amount of label smoothing'
    )
    group.add_argument(
        '-c',
        '--checkpoint-directory',
        type=str,
        default='/tmp/synst/checkpoints',
        help='Where to store model checkpoints'
    )
    group.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10*60,
        help='Generate a checkpoint every `n` seconds'
    )
    group.add_argument(
        '--max-checkpoints',
        type=int,
        default=5,
        help='The maximum number of checkpoints to keep'
    )
    group.add_argument(
        '-e',
        '--max-epochs',
        type=int,
        default=0,
        help='Maximum number of epochs for training the model'
    )
    group.add_argument(
        '--max-steps',
        type=int,
        default=100000,
        help='Maximum number of steps for training the model'
    )
    group.add_argument(
        '-l',
        '--learning-rate',
        dest='base_lr',
        type=float,
        default=None,
        help='The initial learning rate of the optimizer. Defaults to embedding_size ** -0.5'
    )
    group.add_argument(
        '-L',
        '--learning-rate-decay',
        dest='lr_decay',
        type=float,
        default=.999995,
        help='The learning rate decay of the optimizer'
    )
    group.add_argument(
        '--final-learning-rate',
        dest='final_lr',
        type=float,
        default=1e-5,
        help='For the linear annealing schedule'
    )
    group.add_argument(
        '--learning-rate-scheduler',
        dest='lr_scheduler',
        type=str,
        default='warmup',
        choices=['exponential', 'warmup', 'linear','warmup2'],
        help='The learning rate schedule of the optimizer'
    )
    group.add_argument(
        '-w',
        '--warmup-steps',
        type=int,
        default=4000,
        help='Number of warmup steps for the Transformer learning rate'
    )
    group.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'adam-fixed'],
        help='add optimizer'
    )
    group.add_argument(
        '--freeze-layermask',
        default=False,
        action='store_true',
        help="whether to freeze the layermask predictor, this is set to True after training entire model to converge"
    )
    group.add_argument(
        '--linear-tradeoff',
        default=False,
        action='store_true',
        help="use linear scheduled tradeoffs"
    )

    group.add_argument(
        '--sample-interval',
        type=int,
        default=4000,
        help='Number of warmup steps for the Transformer learning rate'
    )
    
    group.add_argument(
        '--sample-times',
        type=int,
        default=500,
        help="number of times to sample layermasks for iterative training"
    )

    group.add_argument(
        '--max-lmp-train-steps',
        type=int,
        default=500,
        help="number of times to train the LMP"
    )

    group.add_argument(
        '--sample-size',
        type=int,
        default=250,
        help="number of examples sampled from valid set"
    )

    group.add_argument(
        '--sample-batch-size',
        type=int,
        default=50,
        help="number of examples sampled from valid set"
    )

    group.add_argument(
        '--step-start-iter-train',
        type=int,
        default=20,
        help="step after which to start training"
    )
    group.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help="debug oracle"
    )

    group.add_argument(
        '--iter-train-lmp-val-n-batches',
        type=int,
        default=2,
        help="training lmp with _n_ valid batches "
    )

    group.add_argument(
        '--iter-train-n-configs', 
        type=int,
        default=100,
        help="train how many configs once"
    )

    group.add_argument(
        '--optimize-the-same', 
        default=False,
        action='store_true',
        help="whether to optimize the same set of configs every validation batch"
    )

    group.add_argument(
        '--max-train-lmp-epochs',
        type=int,
        default=500,
        help='Maximum number of epochs for training the layermask predictor'
    )

    return group

def add_probe_train_args(parser):
    ''' Defines the training specific arguments '''
    group = ArgGroup(parser.add_argument_group('Training'))
    group.add_argument(
        '-A',
        '--accumulate-steps',
        type=int,
        default=1,
        help='How many batches of data to accumulate gradients over'
    )
    group.add_argument(
        '--gold-p',
        type=float,
        default=1.0,
        help='The percentage of time to select gold targets during training (for LSTMs)'
    )
    group.add_argument(
        '--dropout-p',
        type=float,
        default=0.1,
        help='The dropout percentage during training'
    )
    group.add_argument(
        '--early-stopping',
        type=_integer_geq(),
        default=0,
        help='If > 0, stop training after this many checkpoints of increasing nll on the validation'
        ' set. This also implies storing of the best_checkpoint.'
    )
    group.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='The amount of label smoothing'
    )
    group.add_argument(
        '-c',
        '--checkpoint-directory',
        type=str,
        default='/tmp/synst/checkpoints',
        help='Where to store model checkpoints'
    )
    group.add_argument(
        '--stats-directory',
        type=str,
        default='/tmp/synst/stats',
        help='Where to store stats'
    )
    group.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10*60,
        help='Generate a checkpoint every `n` seconds'
    )
    group.add_argument(
        '--max-checkpoints',
        type=int,
        default=5,
        help='The maximum number of checkpoints to keep'
    )
    group.add_argument(
        '-e',
        '--max-epochs',
        type=int,
        default=0,
        help='Maximum number of epochs for training the model'
    )
    group.add_argument(
        '--max-steps',
        type=int,
        default=100000,
        help='Maximum number of steps for training the model'
    )
    group.add_argument(
        '-l',
        '--learning-rate',
        dest='base_lr',
        type=float,
        default=None,
        help='The initial learning rate of the optimizer. Defaults to embedding_size ** -0.5'
    )
    group.add_argument(
        '-L',
        '--learning-rate-decay',
        dest='lr_decay',
        type=float,
        default=.999995,
        help='The learning rate decay of the optimizer'
    )
    group.add_argument(
        '--final-learning-rate',
        dest='final_lr',
        type=float,
        default=1e-5,
        help='For the linear annealing schedule'
    )
    group.add_argument(
        '--learning-rate-scheduler',
        dest='lr_scheduler',
        type=str,
        default='warmup',
        choices=['exponential', 'warmup', 'linear'],
        help='The learning rate schedule of the optimizer'
    )
    group.add_argument(
        '-w',
        '--warmup-steps',
        type=int,
        default=4000,
        help='Number of warmup steps for the Transformer learning rate'
    )

    return group


def add_evaluate_args(parser):
    ''' Defines the evaluation specific arguments '''
    group = ArgGroup(parser.add_argument_group('Evaluation'))
    group.add_argument(
        '--polling',
        default=False,
        action='store_true',
        help='Use a polling observer rather than the default inotify based observer.'
    )
    group.add_argument(
        '--watch-directory',
        type=str,
        default=None,
        help='What directory to watch for new checkpoints.'
        ' If not provided, run a single evaluation using the restore parameter.'
    )

    group.set_defaults(gold_p=0)
    group.set_defaults(dropout_p=0)

    return group


def add_probe_evaluate_args(parser):
    ''' Defines the evaluation specific arguments '''
    group = ArgGroup(parser.add_argument_group('Evaluation'))
    group.add_argument(
        '--polling',
        default=False,
        action='store_true',
        help='Use a polling observer rather than the default inotify based observer.'
    )
    group.add_argument(
        '--watch-directory',
        type=str,
        default=None,
        help='What directory to watch for new checkpoints.'
        ' If not provided, run a single evaluation using the restore parameter.'
    )

    group.add_argument(
        '--stats-directory',
        type=str,
        default='/tmp/synst/stats',
        help='Where to store stats'
    )

    group.set_defaults(gold_p=0)
    group.set_defaults(dropout_p=0)

    return group


def add_translate_args(parser):
    ''' Defines the generation specific arguments '''
    group = ArgGroup(parser.add_argument_group('Generation'))
    group.add_argument(
        '--beam-width',
        default=4,
        type=int,
        help='Default beam width for beam search decoder.'
    )
    group.add_argument(
        '--disable-cache',
        default=False,
        action='store_true',
        help='Whether to disable the use of caching in beam search decoder'
    )
    group.add_argument(
        '--length-penalty',
        type=float,
        default=0.6,
        help='Divides the hypothesis log probabilities in beam search by length^<length penalty>.'
    )
    group.add_argument(
        '--length-basis',
        type=str,
        default=None,
        choices=['input_lens', 'target_lens'],
        help='The basis for max decoding length. Default of None implies no basis, i.e. 0.'
    )
    group.add_argument(
        '--max-decode-length',
        default=50,
        type=int,
        help='How many tokens beyond the length basis to allow decoding to continue.'
    )
    group.add_argument(
        '--output-directory',
        type=str,
        default='/tmp/synst/output',
        help='Where to store translated strings'
    )
    group.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Default output filename is translated_{step}.txt'
    )
    group.add_argument(
        '--order-output',
        default=False,
        action='store_true',
        help='Whether to print the translated strings in the original dataset ordering'
    )
    group.add_argument(
        '--gold-annotations',
        default=False,
        action='store_true',
        help='Whether to use gold annotations rather than have the network predict the annotations'
    )
    group.add_argument(
        '--annotations-only',
        default=False,
        action='store_true',
        help='Whether to only output annotations rather than the predicted translation'
    )
    group.add_argument(
        '--timed',
        type=int,
        default=0,
        const=1,
        nargs='?',
        help='How many times to run translation to gauge the translation speed'
    )

    group.set_defaults(gold_p=0)
    group.set_defaults(dropout_p=0)

    return group


def add_probe_off_diagonal_args(parser):
    ''' Defines the generation specific arguments '''
    group = ArgGroup(parser.add_argument_group('Generation'))
    group.add_argument(
        '--beam-width',
        default=4,
        type=int,
        help='Default beam width for beam search decoder.'
    )
    group.add_argument(
        '--disable-cache',
        default=False,
        action='store_true',
        help='Whether to disable the use of caching in beam search decoder'
    )
    group.add_argument(
        '--length-penalty',
        type=float,
        default=0.6,
        help='Divides the hypothesis log probabilities in beam search by length^<length penalty>.'
    )
    group.add_argument(
        '--length-basis',
        type=str,
        default=None,
        choices=['input_lens', 'target_lens'],
        help='The basis for max decoding length. Default of None implies no basis, i.e. 0.'
    )
    group.add_argument(
        '--max-decode-length',
        default=50,
        type=int,
        help='How many tokens beyond the length basis to allow decoding to continue.'
    )
    group.add_argument(
        '--output-directory',
        type=str,
        default='/tmp/synst/output',
        help='Where to store translated strings'
    )
    group.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Default output filename is translated_{step}.txt'
    )
    group.add_argument(
        '--order-output',
        default=False,
        action='store_true',
        help='Whether to print the translated strings in the original dataset ordering'
    )
    group.add_argument(
        '--gold-annotations',
        default=False,
        action='store_true',
        help='Whether to use gold annotations rather than have the network predict the annotations'
    )
    group.add_argument(
        '--annotations-only',
        default=False,
        action='store_true',
        help='Whether to only output annotations rather than the predicted translation'
    )
    group.add_argument(
        '--timed',
        type=int,
        default=0,
        const=1,
        nargs='?',
        help='How many times to run translation to gauge the translation speed'
    )
    group.add_argument(
        '--off-diagonal-threshold-type',
        type=str,
        default='argmax_offset',
        choices=['argmax_offset', 'argmax_prob', 'argmax_number'],
        help='The type of threshold to determine if the attention of this sentence is considered off-diagonal.'
             'argmax_offset puts sentences whose argmax offset exceed threshold number into off-diagonal category,'
             'argmax_prob puts sentences whose argmax prob among off diagonal words exceed threshould number into off-diagonal category,'
             'argmax_number puts sentences whose number of argmax that is off diagonal exceed the threshold into off-diagonal category.'
             'Only for source attention.'
    )
    group.add_argument(
        '--off-diagonal-threshold-num',
        type=float,
        default=1,
        help='The type of threshold to determine if the attention of this sentence is considered off-diagonal.'
             'Only for source attention'
    )

    group.set_defaults(gold_p=0)
    group.set_defaults(dropout_p=0)

    return group


def add_probe_args(parser):
    ''' Defines the generation specific arguments '''
    group = ArgGroup(parser.add_argument_group('Probe'))
    group.add_argument(
        '--beam-width',
        default=4,
        type=int,
        help='Default beam width for beam search decoder.'
    )
    group.add_argument(
        '--disable-cache',
        default=False,
        action='store_true',
        help='Whether to disable the use of caching in beam search decoder'
    )
    group.add_argument(
        '--length-penalty',
        type=float,
        default=0.6,
        help='Divides the hypothesis log probabilities in beam search by length^<length penalty>.'
    )
    group.add_argument(
        '--length-basis',
        type=str,
        default=None,
        choices=['input_lens', 'target_lens'],
        help='The basis for max decoding length. Default of None implies no basis, i.e. 0.'
    )
    group.add_argument(
        '--max-decode-length',
        default=50,
        type=int,
        help='How many tokens beyond the length basis to allow decoding to continue.'
    )
    group.add_argument(
        '--output-directory',
        type=str,
        default='/tmp/synst/output',
        help='Where to store translated strings'
    )
    group.add_argument(
        '--stats-directory',
        type=str,
        default='/tmp/synst/stats',
        help='Where to store stats'
    )
    group.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Default output filename is translated_{step}.txt'
    )
    group.add_argument(
        '--order-output',
        default=False,
        action='store_true',
        help='Whether to print the translated strings in the original dataset ordering'
    )
    group.add_argument(
        '--gold-annotations',
        default=False,
        action='store_true',
        help='Whether to use gold annotations rather than have the network predict the annotations'
    )
    group.add_argument(
        '--annotations-only',
        default=False,
        action='store_true',
        help='Whether to only output annotations rather than the predicted translation'
    )
    group.add_argument(
        '--timed',
        type=int,
        default=0,
        const=1,
        nargs='?',
        help='How many times to run translation to gauge the translation speed'
    )

    group.set_defaults(gold_p=0)
    group.set_defaults(dropout_p=0)

    return group


def add_pass_args(parser):
    ''' Defines the pass specific arguments '''
    group = ArgGroup(parser.add_argument_group('Pass'))

    group.set_defaults(gold_p=0)
    group.set_defaults(dropout_p=0)

    return group


def parse_args(argv=None):
    ''' Argument parsing '''
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        description='Main entry point for training SynST'
    )
    parser.add_argument(
        '--detect-anomalies',
        default=False,
        action='store_true',
        help='Whether to turn on anomaly detection'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='transformer',
        choices=MODELS,
        help='Which model to instantiate'
    )
    parser.add_argument(
        '-r',
        '--restore',
        type=str,
        default=None,
        help='Location of the checkpoint to restore'
    )
    parser.add_argument(
        '--reset-parameters',
        type=str,
        nargs='*',
        default=[],
        choices=['encoder', 'decoder', 'embeddings', 'step', 'optimizer', 'lr_scheduler'],
        help='What parameters to reset when restoring a checkpoint.'
    )
    parser.add_argument(
        '--average-checkpoints',
        type=int,
        default=1,
        help='How many checkpoints to average over when restoring'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=None,
        help='Set random seed for deterministic evaluation'
    )
    parser.add_argument(
        '--track',
        default=False,
        const=True,
        nargs='?',
        help='Whether to track this experiment. If an experiment id is provided, it will track \
        the existing experiment. If a filename ending with guid it is provided, it will wait \
        until the file exists, then start tracking that experiment.'
    )

    parser.add_argument(
        '--project-name',
        default='probe_transformer',
        type=str,
        help='Specify where to store in comet'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=0,
        action='count',
        help='Increase the verbosity level'
    )

    groups = {}
    groups['cuda'] = add_cuda_args(parser)
    groups['data'] = add_data_args(parser)

    model_groups = {}
    model_groups['transformer'] = add_transformer_args(parser)
    model_groups['probe_transformer'] = add_probe_transformer_args(parser)
    model_groups['new_transformer'] = add_new_transformer_args(parser)
    model_groups['parse_transformer'] = add_parse_transformer_args(parser)
    model_groups['probe_new_transformer'] = add_new_transformer_args(parser)

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='Train a model')
    groups['train'] = add_train_args(train_parser)
    train_parser.set_defaults(
        action=Trainer,
        action_type='train',
        action_config=groups['train'],
        shuffle=True
    )

    iterative_train_parser = subparsers.add_parser('iterative_train', help='Train a model iteratively with sampled layermasks')
    groups['iterative_train'] = add_iterative_train_args(iterative_train_parser)
    iterative_train_parser.set_defaults(
        action=IterativeTrainer,
        action_type='iterative_train',
        action_config=groups['iterative_train'],
        shuffle=True
    )

    probe_train_parser = subparsers.add_parser('probe_train', help='Train a model while probing')
    groups['probe_train'] = add_probe_train_args(probe_train_parser)
    probe_train_parser.set_defaults(
        action=ProbeTrainer,
        action_type='probe_train',
        action_config=groups['probe_train'],
        shuffle=True
    )

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    groups['evaluate'] = add_evaluate_args(evaluate_parser)
    evaluate_parser.set_defaults(
        action=Evaluator,
        action_type='evaluate',
        action_config=groups['evaluate'],
        shuffle=False
    )

    probe_evaluate_parser = subparsers.add_parser('probe_evaluate', help='Probe Evaluate a model')
    groups['probe_evaluate'] = add_probe_evaluate_args(probe_evaluate_parser)
    probe_evaluate_parser.set_defaults(
        action=ProbeEvaluator,
        action_type='probe_evaluate',
        action_config=groups['probe_evaluate'],
        shuffle=False
    )

    translate_parser = subparsers.add_parser('translate', help='Translate from a model')
    groups['translate'] = add_translate_args(translate_parser)
    translate_parser.set_defaults(
        action=Translator,
        action_type='translate',
        action_config=groups['translate'],
        shuffle=False
    )

    probe_new_translate_parser = subparsers.add_parser('probe_new_translate', help='Probe New Translate from a model')
    groups['probe_new_translate'] = add_translate_args(probe_new_translate_parser)
    probe_new_translate_parser.set_defaults(
        action=ProbeNewTranslator,
        action_type='probe_new_translate',
        action_config=groups['probe_new_translate'],
        shuffle=False
    )

    probe_off_diagonal_parser = subparsers.add_parser('probe_off_diagonal', help='Probe the performance of sentences where learned attention looks off diagonal')
    groups['probe_off_diagonal'] = add_probe_off_diagonal_args(probe_off_diagonal_parser)
    probe_new_translate_parser.set_defaults(
        action=ProbeOffDiagonal,
        action_type='probe_off_diagonal',
        action_config=groups['probe_off_diagonal'],
        shuffle=False
    )

    probe_parser = subparsers.add_parser('probe', help='Probe a model')
    groups['probe'] = add_probe_args(probe_parser)
    probe_parser.set_defaults(
        action=Prober,
        action_type='probe',
        action_config=groups['probe'],
        shuffle=False
    )

    pass_parser = subparsers.add_parser('pass', help='No action... Useful to invoke preprocessing')
    groups['pass'] = add_pass_args(pass_parser)
    pass_parser.set_defaults(
        action=Pass,
        action_type='pass',
        action_config=groups['pass'],
        shuffle=False
    )

    args = parser.parse_args(args=argv)

    args.config = SimpleNamespace()
    for group_name, group in groups.items():
        setattr(args.config, group_name, group.read(args))

    model_group = model_groups[args.model]
    args.config.model = model_group.read(args)
    args.config.model.gold_p = args.action_config.gold_p
    args.config.model.dropout_p = args.action_config.dropout_p
    args.config.model.label_smoothing = args.config.train.label_smoothing
    if hasattr(args.config.train, 'base_lr') and not args.config.train.base_lr:
        args.config.train.base_lr = model_group.embedding_size ** -0.5

    if hasattr(args.config.iterative_train, 'base_lr') and not args.config.iterative_train.base_lr:
        args.config.iterative_train.base_lr = model_group.embedding_size ** -0.5

    args.version = get_version_string()
    if args.track and '-dirty' in args.version:
        raise RuntimeError('''
Trying to track an experiment, but the workspace is dirty!
Commit your changes first, then try again.''')

    api_key = None if args.track else ''
    experiment_type = Experiment
    experiment_args = [api_key]
    if isinstance(args.track, str):
        experiment_type = ExistingExperiment
        if args.track.endswith('.guid'):
            wait_count = 0
            while not os.path.exists(args.track):
                wait_string = '...'[:wait_count % 4]
                wait_count += 1

                print(f'\r\033[KWaiting for experiment: {args.track} {wait_string}', end='')
                time.sleep(1)

            print(f'\r\033[KLoading experiment: {args.track}')
            with open(args.track, 'rt') as guid_file:
                experiment_args.append(guid_file.readline().strip())
        else:
            experiment_args.append(args.track)

    args.experiment = experiment_type(
        *experiment_args,
        project_name=args.project_name,
        workspace='umass-nlp',
        disabled=not args.track,
        auto_metric_logging=False,
        auto_output_logging=None,
        auto_param_logging=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_graph=False,
        log_code=False,
        parse_args=False,
    )

    if args.track and experiment_type == Experiment and args.action_type == 'train':
        with open(os.path.join(args.checkpoint_directory, 'experiment.guid'), 'wt') as guid_file:
            guid_file.write(args.experiment.id)

    # This needs to be called separately to disable monkey patching of the ML frameworks which is on
    # by default :(
    args.experiment.disable_mp()

    if experiment_type is Experiment:
        args.experiment.log_parameter('version', args.version)
        args.experiment.log_parameters(
            args.config.data.dict, prefix='data'
        )
        args.experiment.log_parameters(
            args.config.model.dict, prefix='model'
        )
        args.experiment.log_parameters(
            args.action_config.dict, prefix=args.action_type
        )

    if not args.config.cuda.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.num_devices = torch.cuda.device_count()
    else:
        args.device = torch.device('cpu')
        args.num_devices = 1

    args.model = MODELS[args.model]
    args.config.data.dataset = DATASETS[args.dataset]
    args.config.data.span = args.config.model.span

    if args.seed is not None:
        args.seed_fn = get_random_seed_fn(args.seed)
        args.seed_fn()
    else:
        args.seed_fn = None

    if args.action_type == 'evaluate':
        args.action_config.average_checkpoints = args.average_checkpoints

    if (args.action_type == 'translate' or args.action_type == 'probe_new_translate') and args.num_devices > 1:
        # Caching is currently not thread-safe
        args.action_config.disable_cache = True

    return args
