'''
A module which implements the basic Transformer
'''
import uuid
import threading
import pdb
import torch
import numpy as np
import random
from torch import nn

from models.new_attention import NewAttention
from models.attention import MultiHeadedAttention
from models.embeddings import PositionEmbedding, TokenEmbedding
from models.utils import LabelSmoothingLoss, Translator
from utils import left_shift, right_shift, triu
from torch.distributions import Bernoulli
import torch.nn.functional as F
from itertools import combinations
from torch.nn import BCELoss


class TransformerSublayer(nn.Module):
    '''
    Implements a sub layer of the transformer model, which consists of:
    1) A sub layer module
    2) Followed by dropout
    3) Plus a residual connection
    4) With layer normalization
    '''
    def __init__(self, sublayer, sublayer_shape, dropout_p=0.1):
        ''' Initialize the transformer sublayer '''
        super(TransformerSublayer, self).__init__()

        self.sublayer = sublayer
        self.norm = nn.LayerNorm(sublayer_shape)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        self.norm.reset_parameters()

    def forward(self, inputs, gating_weight, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        out_dropout = self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs))
        ret = self.norm(inputs + gating_weight[:, None, None] * out_dropout)
        skip = inputs * (1-gating_weight)[:, None, None]
        ret = gating_weight[:, None, None] * ret + skip
        return ret

class TransformerFFN(nn.Module):
    ''' Implements the Transformer feed-forward network '''
    def __init__(self, embedding_size, hidden_dim):
        super(TransformerFFN, self).__init__()

        self.relu = nn.ReLU()

        self.hidden = nn.Linear(embedding_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, embedding_size)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.hidden.weight, gain)
        nn.init.constant_(self.hidden.bias, 0.)

        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.output.weight, gain)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass of the feed-forward network '''
        return self.output(self.relu(self.hidden(inputs)))


class TransformerEncoderLayer(nn.Module):
    ''' Implements a single encoder layer in a transformer encoder stack '''
    def __init__(self, attn_config, num_heads, dim, hidden_dim, layer_i, dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerEncoderLayer, self).__init__()
        self.no_attn = attn_config['no_attn']

        if attn_config['ffn_layer'][layer_i]:
            self.ffn = TransformerSublayer(
                TransformerFFN(dim, hidden_dim),
                dim, dropout_p
            )
            print('enc layer %i has ffn' % layer_i)

        if not self.no_attn:
            self.self_attention = TransformerSublayer(
                NewAttention(attn_config, dim, num_heads),
                dim, dropout_p
            )

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()

    def forward(self, inputs, layer_i, word_embedding, gating_weight=1):  # pylint:disable=arguments-differ
        ''' The forward pass '''
        mask = inputs['mask']
        state = inputs['state']

        # print("encoder self attention")

        if not self.no_attn:
            state = self.self_attention(
                state, gating_weight, # residual
                state, state, state, mask,  # passed to multiheaded attention
                layer_i=layer_i, word_embedding=word_embedding
            )

        if hasattr(self, 'ffn'):
            state = self.ffn(
                state, gating_weight,  # residual
                state, # passed to feed-forward network  
            )

        return {'state': state, 'mask': mask}


class TransformerDecoderLayer(nn.Module):
    ''' Implements a single decoder layer in a transformer decoder stack '''
    def __init__(self, dec_attn_config, enc_dec_attn_config, num_heads, dim, hidden_dim, layer_i, causal=True, span=1,
                 dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerDecoderLayer, self).__init__()

        self.span = span
        self.causal = causal
        self.uuid = uuid.uuid4()

        self.enc_dec_attn_config = enc_dec_attn_config
        self.no_attn = dec_attn_config['no_attn']


        if dec_attn_config['ffn_layer'][layer_i]:
            self.ffn = TransformerSublayer(
                TransformerFFN(dim, hidden_dim),
                dim, dropout_p
            )
            print('dec layer %i has ffn' % layer_i)

        if not self.no_attn:
            self.self_attention = TransformerSublayer(
                NewAttention(dec_attn_config, dim, num_heads),
                dim, dropout_p
            )

        if self.enc_dec_attn_config['enc_dec_attn_layer'] == 1 or \
                (type(self.enc_dec_attn_config['enc_dec_attn_layer'] is list) and
                 self.enc_dec_attn_config['enc_dec_attn_layer'][layer_i] == 1):
            if self.enc_dec_attn_config['enc_dec_attn_num_heads'] == -1:
                src_num_heads = num_heads
            elif type(self.enc_dec_attn_config['enc_dec_attn_num_heads']) is not list:
                src_num_heads = self.enc_dec_attn_config['enc_dec_attn_num_heads']
            else:
                src_num_heads = self.enc_dec_attn_config['enc_dec_attn_num_heads'][layer_i]
            assert src_num_heads != 0

            self.source_attention = TransformerSublayer(
                NewAttention(enc_dec_attn_config, dim, src_num_heads),
                dim, dropout_p
            )

            print('layer %i num of src heads %i' % (layer_i, src_num_heads))

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()
        if hasattr(self, 'source_attention'):
            self.source_attention.reset_parameters()

    def forward(self, inputs, sources, layer_i, word_embedding, gating_weight=1): # pylint:disable=arguments-differ
        ''' The forward pass '''
        mask = inputs['mask']
        state = inputs['state']
        cache = inputs.get('cache')
        input_lens = inputs.get('input_lens')

        decoder_position = state.shape[1] - 1

        if not self.no_attn:
            kwargs = {'layer_i': layer_i}
            if self.causal and cache is not None:
                # If caching, only want the last k=span sequence values. Requires no causal masking.
                residual = state[:, -self.span:]
                kwargs['num_queries'] = self.span
                kwargs['decoder_position'] = decoder_position
                kwargs['word_embedding'] = word_embedding[:, -self.span:]
            else:
                # If not caching, use the full sequence and ensure an appropriate causal mask
                residual = state
                kwargs['key_mask'] = mask
                kwargs['attention_mask'] = self.mask(state)
                kwargs['word_embedding'] = word_embedding

            # print("decoder self attention")

            state = self.self_attention(
                residual, gating_weight, # residual
                state, state, state, **kwargs # passed to multiheaded attention
            )
        else:
            if self.causal and cache is not None:
                state = state[:, -self.span:]

        source = sources['state']
        # print("source", source)
        kwargs = {'key_mask': sources['mask'], 'layer_i': layer_i, 'input_lens': input_lens}
        if self.causal and cache is not None:
            kwargs['num_queries'] = self.span
            kwargs['decoder_position'] = decoder_position
            kwargs['word_embedding'] = word_embedding[:, -self.span:]
        else:
            kwargs['word_embedding'] = word_embedding

        # print("decoder source attention")

        if hasattr(self, 'source_attention'):
            # print("in source, state", state.shape)
            state = self.source_attention(
                state, gating_weight, # residual
                source, source, state, **kwargs # passed to multiheaded attention
            )

        if hasattr(self, 'ffn'):
            state = self.ffn(
                state, gating_weight, # residual
                state, # passed to feed-forward network
            )

        if self.causal and cache is not None:
            cached = cache.get(self.uuid)
            if cached is None:
                cache[self.uuid] = state
            else:
                # print("cached", cached.shape)
                # print("state", state.shape)
                try:
                    state = cache[self.uuid] = torch.cat((cached, state), 1)
                except:
                    pdb.set_trace()

        return {'state': state, 'mask': mask, 'cache': cache}

    _masks = threading.local()
    def mask(self, inputs):
        '''
        Get a self-attention mask
        The mask will be of shape [T x T] containing elements from the set {0, -inf}
        Input shape:  (B x T x E)
        Output shape: (T x T)
        '''
        if not self.causal:
            return None

        dim = inputs.shape[1]
        device = inputs.device
        mask_store = TransformerDecoderLayer._masks.__dict__
        if device not in mask_store:
            mask = inputs.new_full((dim, dim), float('-inf'))
            mask_store[device] = triu(mask, 1, self.span, self.span)

        mask = mask_store[device]
        if mask.shape[0] < dim:
            mask = mask.resize_(dim, dim).fill_(float('-inf'))
            mask_store[device] = triu(mask, 1, self.span, self.span)
            mask = mask_store[device]

        return mask[None, :dim, :dim]

class LayerMaskPredictor(nn.Module):
    def __init__(self, embedding_size, 
                       hidden_size, 
                       num_layers, 
                       action_type, 
                       lmp_type, 
                       noisy,
                       allon_threshold,
                       potential_threshold,
                       shuffle_configs,
                       num_configs,
                       loss_func,
                       lmp_eval_mode,
                       dropout_p=0.1):
        super(LayerMaskPredictor, self).__init__()
        self.num_layers = num_layers

        #### for debugging oracle
        num_layer = 2 * num_layers
        all_combs = sum([list(combinations(range(num_layer), k)) for k in range(1, num_layer+1)], [])
        all_combs = [x for x in all_combs if any(y >= num_layer//2 for y in x)]
        print("shuffle configs", shuffle_configs)
        if shuffle_configs:
            random.Random(42).shuffle(all_combs)

        all_combs = all_combs[:num_configs]
        if num_configs != 4032:
            print("append all-on config")
            all_combs.append(tuple(i for i in range(num_layer)))

        self.all_configs = torch.zeros(len(all_combs), num_layer, device=torch.device("cuda"))
        for ci, c in enumerate(all_combs):
            for cii in c:
                self.all_configs[ci, cii] += 1


        """
        To be deleted later, adding this because oracle result has label shift
        e.g. [0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1.] change to [1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1.]
        1. first layer always 1
        2. decoder layer shift backward by 1
        """
        self.all_configs[:, 0] = 1
        self.all_configs[:, self.num_layers:] = self.all_configs[:, self.num_layers-1:-1]
        """
        To be deleted
        """


        self.ci_allon = all_combs.index(tuple(i for i in range(num_layer)))
        self.all_configs_sum_layer = self.all_configs.sum(dim=1) # len(all_combs) x 1
        print("all-on config ci: %i" % self.ci_allon)

        # config range: while training with samples, only optimize certain range; during test time, need to know which range is optimized
        self.config_start = 0
        self.config_end = -1
        self.allon_threshold = allon_threshold
        self.potential_threshold = potential_threshold
        print("potential threshold %f" % self.potential_threshold)

        if lmp_type not in ["random"]:
            if not noisy:
                if lmp_type == "iterative_training":
                    self.proj1 = nn.Linear(embedding_size, 2 * num_layers)
                elif lmp_type == "iterative_training_debug_oracle":
                    self.proj1 = nn.Linear(embedding_size, len(all_combs)-1)
                else:
                    self.proj1 = nn.Linear(embedding_size, hidden_size)
                    self.proj2 = nn.Linear(hidden_size, 2 * num_layers)
            else:
                self.proj1 = nn.Linear(embedding_size, 2 * num_layers)
                self.proj_noise = nn.Linear(embedding_size, 2 * num_layers)

            self.dropout = nn.Dropout(dropout_p, inplace=True)

            self.action_type = action_type
            self.lmp_type = lmp_type
            self.noisy=noisy

            self.bce_loss = BCELoss(reduction='none')

            self.reset_parameters()
        else:
            self.sample_distribution = torch.ones(2 * num_layers, device=torch.device("cuda")) * 0.5 # init 0.5

        if self.lmp_type == "iterative_training_debug_oracle":
            self.loss_func = loss_func
        self.eval = lmp_eval_mode
        
    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        gain = nn.init.calculate_gain('linear')

        nn.init.xavier_uniform_(self.proj1.weight, gain)
        nn.init.constant_(self.proj1.bias, 0.)
        if not self.noisy:
            if "iterative_training" not in self.lmp_type:
                nn.init.xavier_uniform_(self.proj2.weight, gain)
                nn.init.constant_(self.proj2.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.proj_noise.weight, gain)
            nn.init.constant_(self.proj_noise.bias, 0.)

    def forward(self, lmp_input, lmp_input_mask, aggregate_stats=None):
        '''
            lmp_input: [bs, L, embedding_size]
            layermask: [bs, 2*num_layers]
            return: sampled layermask, raw-layermask-distribution
        '''
        
        # not skipping
        if self.lmp_type == "noskip":
            return None, torch.ones(lmp_input.size(0), self.num_layers * 2, device=torch.device("cuda"))

        # use lmp
        if not self.eval:
            if self.loss_func == 'binary_cls':
                lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
                layermask = self.proj1(torch.mean(lmp_input,1))
                layermask = torch.sigmoid(layermask)
                loss = self.bce_loss(layermask, aggregate_stats)
                loss = loss.mean(dim=1).mean()

                return loss, None

            elif self.loss_func == "regr":
                lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
                layermask = self.proj1(torch.mean(lmp_input,1))
                layermask = torch.sigmoid(layermask)
                loss = ((layermask - aggregate_stats)**2).mean(dim=1).mean()
                return loss, None

            elif self.loss_func == "scaled_regr":
                lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
                layermask = self.proj1(torch.mean(lmp_input,1))
                layermask = torch.sigmoid(layermask)
                loss = ((layermask - aggregate_stats)**2).mean(dim=1).mean()
                return loss, None

            elif self.loss_func == "rank":
                pass

            else:
                pass

        # testing
        else:
            bs, L, emb = lmp_input.shape
            print('using layermask predictor')

            # assert aggregate_stats is None

            if self.loss_func == 'binary_cls':
                
                lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
                layermask = self.proj1(torch.mean(lmp_input,1))
                layermask = torch.sigmoid(layermask)

                if aggregate_stats is not None:
                    loss = self.bce_loss(layermask, aggregate_stats)
                    loss = loss.mean(dim=1).mean()
                    return loss, None

                ret = torch.zeros(layermask.shape[0], self.num_layers * 2, device=torch.device("cuda"))
                max_val, _ = layermask.max(dim=1)
                # filter configs within range (max-potential_threshold, max)
                filtered = (layermask + self.potential_threshold >= max_val[:, None]).float() * self.all_configs_sum_layer[:-1]
                filtered[filtered == 0] = float("inf")
                _, ci = torch.min(filtered, dim=1)
                ci_val = layermask[range(bs), ci]
                # ci[ci_val < self.allon_threshold] = self.ci_allon
                print("{:.2f} {:.2f} {:.2f} {:.2f}".format(ci_val.mean().item(), ci_val.max().item(), ci_val.min().item(), max_val.mean().item() - 2*self.potential_threshold))
                ci[ci_val < max_val.mean().item() - 2*self.potential_threshold] = self.ci_allon
                ret = self.all_configs[ci]

                return None, ret

            elif self.loss_func == "regr":

#                lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
                layermask = self.proj1(torch.mean(lmp_input,1))
                layermask = torch.sigmoid(layermask)

                if aggregate_stats is not None:
                    loss = ((layermask - aggregate_stats)**2).mean(dim=1).mean()
                    return loss, None

                ret = torch.zeros(layermask.shape[0], self.num_layers * 2, device=torch.device("cuda"))
                max_val, _ = layermask.max(dim=1)
                # filter configs within range (max-potential_threshold, max)
                filtered = (layermask + self.potential_threshold >= max_val[:, None]).float() * self.all_configs_sum_layer[:-1]
                filtered[filtered == 0] = float("inf")
                _, ci = torch.min(filtered, dim=1)
                ci_val = layermask[range(bs), ci]

                print("{:.2f} {:.2f} {:.2f} {:.2f}".format(ci_val.mean().item(), ci_val.max().item(), ci_val.min().item(), max_val.mean().item() - 2*self.potential_threshold))
                # ci[ci_val < self.allon_threshold] = self.ci_allon

                ci[ci_val < max_val.mean().item() - 2*self.potential_threshold] = self.ci_allon
                ret = self.all_configs[ci]

                return None, ret

            elif self.loss_func == "scaled_regr":
                lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
                layermask = self.proj1(torch.mean(lmp_input,1))
                layermask = torch.sigmoid(layermask)
                if aggregate_stats is not None:
                    loss = ((layermask - aggregate_stats)**2).mean(dim=1).mean()
                    return loss, None
                ret = torch.zeros(layermask.shape[0], self.num_layers * 2, device=torch.device("cuda"))
                max_val, _ = layermask.max(dim=1)
                # filter configs within range (max-potential_threshold, max)
                filtered = (layermask + self.potential_threshold >= max_val[:, None]).float() * self.all_configs_sum_layer[:-1]
                filtered[filtered == 0] = float("inf")
                _, ci = torch.min(filtered, dim=1)
                ci_val = layermask[range(bs), ci]
                print("{:.2f} {:.2f} {:.2f} {:.2f}".format(ci_val.mean().item(), ci_val.max().item(), ci_val.min().item(), max_val.mean().item() - 2*self.potential_threshold))
                # ci[ci_val < self.allon_threshold] = self.ci_allon
                ci[ci_val < max_val.mean().item() - 2*self.potential_threshold] = self.ci_allon
                ret = self.all_configs[ci]

                return None, ret

            elif self.loss_func == "rank":
                pass

            else:
                pass


    def random_layermask_sampling(self):

        # sample from distribution
        all_sample = Bernoulli(torch.ones(config.num_layers * 2) * self.sample_distribution).sample()
        # must have a decoder
        if not any(all_sample[self.num_layers:self.num_layers*2]):
            all_sample[np.random.randint(self.num_layers, self.num_layers*2)] = 1

        return all_sample


class NewTransformer(nn.Module):
    ''' The New Transformer module '''
    def __init__(self, config, dataset):
        ''' Initialize the Transformer '''
        super(NewTransformer, self).__init__()

        self.dataset = dataset
        self.span = config.span
        self.embedding = TokenEmbedding(
            dataset.vocab_size,
            config.embedding_size,
            padding_idx=self.padding_idx
        )
        self.position_embedding = PositionEmbedding(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

        # Allow for overriding the encoders and decoders in dervied classes
        self.encoders = type(self).create_encoders(config)
        self.decoders = self.create_decoders(config)

        self.label_smoothing = LabelSmoothingLoss(
            config.label_smoothing or 0,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction='none'
        )


        self.layermask_type = config.layermask_type
        self.gating_tradeoff = config.gating_tradeoff
        self.diversity_tradeoff = config.diversity_tradeoff

        self.random_layermask_p = 0.5

        self.layermask = None

        # layermask predictor
        self.layer_mask_predictor = LayerMaskPredictor(config.embedding_size, 
                                                       config.hidden_dim, 
                                                       config.num_layers, 
                                                       config.action_type, 
                                                       config.layermask_type,
                                                       config.layermask_noisy,
                                                       config.allon_threshold,
                                                       config.potential_threshold,
                                                       config.shuffle_lmp_configs,
                                                       config.num_configs,
                                                       config.loss_func,
                                                       config.lmp_eval_mode)


    @classmethod
    def create_encoders(cls, config):
        ''' Create the transformer encoders '''
        kwargs = {'dropout_p': config.dropout_p}

        if config.ffn_layer == -1:
            config.ffn_layer = [1] * config.num_layers
        assert len(config.ffn_layer) == config.num_layers

        attn_config = {'attn_type': config.attn_type,
                       'attn_position': config.attn_position,
                       'attn_param': config.attn_param,
                       'attn_displacement': config.attn_displacement,
                       'num_layers': config.num_layers,
                       'num_heads': config.num_heads,
                       'attn_concat': config.attn_concat,
                       'which_attn': 'encoder',
                       'attn_weights': config.attn_weights,
                       'attn_score': config.attn_score,
                       'attn_bins': config.attn_bins,
                       'attn_threshold': config.attn_threshold,
                       'attn_window': config.attn_window,
                       'attn_indexing': config.enc_attn_indexing,
                       'no_attn': config.enc_no_attn,
                       'indexing_type': config.indexing_type,
                       'ffn_layer': config.ffn_layer}
        args = [attn_config, config.num_heads, config.embedding_size, config.hidden_dim]

        encoders = nn.ModuleList([
            TransformerEncoderLayer(*args, layer_i, **kwargs)
            for layer_i in range(config.num_layers)
        ])

        if config.tie_ffn_weights:
            for enc in encoders[1:]:
                if hasattr(enc, 'ffn') and hasattr(encoders[0], 'ffn'):
                    enc.ffn.sublayer.hidden.weight = encoders[0].ffn.sublayer.hidden.weight
                    enc.ffn.sublayer.output.weight = encoders[0].ffn.sublayer.output.weight
                elif hasattr(enc, 'ffn') and hasattr(encoders[1], 'ffn'):
                    enc.ffn.sublayer.hidden.weight = encoders[1].ffn.sublayer.hidden.weight
                    enc.ffn.sublayer.output.weight = encoders[1].ffn.sublayer.output.weight

        return encoders

    # @classmethod
    def create_decoders(self, config):
        ''' Create the transformer decoders '''
        kwargs = {'dropout_p': config.dropout_p, 'span': config.span}

        if config.ffn_layer == -1:
            config.ffn_layer = [1] * config.num_layers
        assert len(config.ffn_layer) == config.num_layers

        dec_attn_config = {'attn_type': config.dec_attn_type,
                           'attn_position': config.dec_attn_position,
                           'attn_param': config.dec_attn_param,
                           'attn_displacement': config.dec_attn_displacement,
                           'num_layers': config.num_layers,
                           'num_heads': config.num_heads,
                           'attn_concat': config.dec_attn_concat,
                           'which_attn': 'decoder',
                           'attn_weights': config.dec_attn_weights,
                           'attn_score': config.dec_attn_score,
                           'attn_bins': config.dec_attn_bins,
                           'attn_threshold': config.dec_attn_threshold,
                           'attn_window': config.dec_attn_window,
                           'attn_indexing': config.dec_attn_indexing,
                           'no_attn': config.dec_no_attn,
                           'indexing_type': config.indexing_type,
                           'ffn_layer': config.ffn_layer}
        enc_dec_attn_config = {'attn_type': config.enc_dec_attn_type,
                               'attn_position': config.enc_dec_attn_position,
                               'attn_param': config.enc_dec_attn_param,
                               'attn_displacement': config.enc_dec_attn_displacement,
                               'num_layers': config.num_layers,
                               'num_heads': config.num_heads,
                               'word_count_ratio': self.dataset.word_count_ratio,
                               'attn_concat': config.enc_dec_attn_concat,
                               'which_attn': 'source',
                               'attn_weights': config.enc_dec_attn_weights,
                               'attn_score': config.enc_dec_attn_score,
                               'attn_bins': config.enc_dec_attn_bins,
                               'enc_dec_attn_layer': config.enc_dec_attn_layer,
                               'enc_dec_attn_num_heads': config.enc_dec_attn_num_heads,
                               'attn_threshold': config.enc_dec_attn_threshold,
                               'attn_window': config.enc_dec_attn_window,
                               'attn_indexing': config.enc_dec_attn_indexing,
                               'indexing_type': config.indexing_type,
                               'ffn_layer': config.ffn_layer
                               }
        args = [dec_attn_config, enc_dec_attn_config, config.num_heads, config.embedding_size, config.hidden_dim]
        decoders = nn.ModuleList([
            TransformerDecoderLayer(*args, layer_i, **kwargs)
            for layer_i in range(config.num_layers)
        ])

        if config.tie_ffn_weights:
            for dec in decoders[1:]:
                if hasattr(dec, 'ffn') and hasattr(decoders[0], 'ffn'):
                    dec.ffn.sublayer.hidden.weight = decoders[0].ffn.sublayer.hidden.weight
                    dec.ffn.sublayer.output.weight = decoders[0].ffn.sublayer.output.weight
                elif hasattr(dec, 'ffn') and hasattr(decoders[1], 'ffn'):
                    dec.ffn.sublayer.hidden.weight = decoders[1].ffn.sublayer.hidden.weight
                    dec.ffn.sublayer.output.weight = decoders[1].ffn.sublayer.output.weight

        return decoders


    @property
    def sos_idx(self):
        ''' Return the sos index '''
        return self.dataset.sos_idx

    @property
    def padding_idx(self):
        ''' Return the padding index '''
        return self.dataset.padding_idx

    def translator(self, config):
        ''' Get a translator for this model '''
        return Translator(config, self, self.dataset)

    def reset_named_parameters(self, modules):
        ''' Get a translator for this model '''
        if 'encoder' in modules:
            for encoder in self.encoders:
                encoder.reset_parameters()
        if 'decoder' in modules:
            for decoder in self.decoders:
                decoder.reset_parameters()
        if 'embeddings' in modules:
            self.embedding.reset_parameters()

    def forward(self, batch, step_progress=0): # pylint:disable=arguments-differ
        ''' A batch of inputs and targets '''
        """
            step_progress = curr_step / max-step
        """

        encoded, _, raw_layermask = self.encode(batch['inputs'])
        self.layermask = raw_layermask

        decoded = self.decode(
            encoded,
            right_shift(right_shift(batch['targets']), shift=self.span - 1, fill=self.sos_idx),
            input_lens=batch['input_lens'],
            raw_layermask=raw_layermask
        )

        logits = decoded['logits']
        dims = list(range(1, logits.dim()))
        targets = left_shift(batch['targets'])
        nll = self.cross_entropy(logits, targets).sum(dims[:-1])
        smoothed_nll = self.label_smoothing(logits, targets).sum(dims)

        sum_layermask = raw_layermask.sum(dim=1) # [bs, ]

        loss = smoothed_nll

        return loss, nll, None, sum_layermask.mean()

    def encode(self, inputs, raw_layermask=None):
        ''' Encode the inputs '''
        word_embedding = self.embed(inputs, self.embedding)
        encoded = {
            'state': word_embedding,
            'mask': inputs.eq(self.padding_idx)
        }

        print(self.layermask_type)

        if raw_layermask is None:
            layer_mask, lmp_raw_layermask = self.layer_mask_predictor(encoded['state'], encoded['mask'])
            raw_layermask = lmp_raw_layermask
        else:
            layer_mask = None
        for i, encoder in enumerate(self.encoders):
            encoded = encoder(encoded, i, word_embedding, gating_weight=raw_layermask[:, i])
                
        return encoded, layer_mask, raw_layermask

    def decode(self, encoded, targets, decoders=None, 
                                       embedding=None, 
                                       cache=None, 
                                       mask=None, 
                                       input_lens=None, 
                                       raw_layermask=None):
        ''' Decode the encoded sequence to the targets '''
        if decoders is None:
            decoders = self.decoders

        if embedding is None:
            embedding = self.embedding

        word_embedding = self.embed(targets, embedding)

        decoded = {
            'cache': cache,
            'state': word_embedding,
            'mask': targets.eq(self.padding_idx) if mask is None else mask,
            'input_lens': input_lens
        }

        if len(raw_layermask) != encoded['state'].shape[0]:
            pdb.set_trace()

        for i, decoder in enumerate(decoders):
            decoded = decoder(decoded, encoded, i, word_embedding, gating_weight=raw_layermask[:, len(decoders) + i])
           
        # compute projection to the vocabulary
        state = decoded['state']
        if cache is not None:
            state = state[:, -self.span:]

        return {
            'cache': decoded.get('cache'),
            'logits': embedding(state, transpose=True).transpose(2, 1),  # transpose to B x C x ...
        }

    def embed(self, inputs, token_embedding):
        ''' Embed the given inputs '''
        return self.dropout(token_embedding(inputs) + self.position_embedding(inputs))


    def set_LMP_type(self, lmp_type):
        if lmp_type not in ["random", "noskip", 'iterative_training', 'iterative_training_debug_oracle']:
            print("cannot set to this type after initiliazation!")
            exit(-1)
        self.layer_mask_predictor.lmp_type = lmp_type
        self.layermask_type = lmp_type

    def set_LMP_config_range(self, start, end):
        
        self.layer_mask_predictor.config_start = start
        self.layer_mask_predictor.config_end = end
