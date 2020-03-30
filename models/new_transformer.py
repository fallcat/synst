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
from models.layer_mask_predictor import LayerMaskPredictor


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

    def forward(self, inputs, gating_weight, ensemble, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        if not ensemble and len(gating_weight.size()) == 0:
            return self.norm(inputs + self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs)))
        else:
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
    def __init__(self, attn_config, num_heads, dim, hidden_dim, layer_i, dropout_p=0.1, ensemble=False):
        ''' Initialize the transformer layer '''
        super(TransformerEncoderLayer, self).__init__()
        self.no_attn = attn_config['no_attn']
        self.ensemble = ensemble

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
                state, gating_weight, self.ensemble, # residual
                state, state, state, mask,  # passed to multiheaded attention
                layer_i=layer_i, word_embedding=word_embedding
            )

        if hasattr(self, 'ffn'):
            state = self.ffn(
                state, gating_weight, self.ensemble, # residual
                state, # passed to feed-forward network  
            )

        return {'state': state, 'mask': mask}


class TransformerDecoderLayer(nn.Module):
    ''' Implements a single decoder layer in a transformer decoder stack '''
    def __init__(self, dec_attn_config, enc_dec_attn_config, num_heads, dim, hidden_dim, layer_i, layermasks_len, causal=True, span=1,
                 dropout_p=0.1, ensemble=False):
        ''' Initialize the transformer layer '''
        super(TransformerDecoderLayer, self).__init__()

        self.span = span
        self.causal = causal
        self.uuid = uuid.uuid4()

        self.enc_dec_attn_config = enc_dec_attn_config
        self.no_attn = dec_attn_config['no_attn']
        self.layermasks_len = layermasks_len
        self.ensemble = ensemble

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
        pdb.set_trace()
        mask = inputs['mask']
        state = inputs['state']
        cache = inputs.get('cache')
        input_lens = inputs.get('input_lens')
        print("1 state ", state.shape)

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
                residual, gating_weight, self.ensemble, # residual
                state, state, state, **kwargs # passed to multiheaded attention
            )
        else:
            if self.causal and cache is not None:
                state = state[:, -self.span:]

        print("2 state ", state.shape)

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
                state, gating_weight, self.ensemble, # residual
                source, source, state, **kwargs # passed to multiheaded attention
            )

        if hasattr(self, 'ffn'):
            state = self.ffn(
                state, gating_weight, self.ensemble, # residual
                state, # passed to feed-forward network
            )

        print("3 state ", state.shape)

        if self.causal and cache is not None:
            cached = cache.get(self.uuid)
            if self.ensemble:
                state_to_cache = state.view(-1, self.layermasks_len, state.shape[1], state.shape[2]).mean(1)
            else:
                state_to_cache = state
            if cached is None:
                # pdb.set_trace()
                cache[self.uuid] = state_to_cache
            else:
                # print("cached", cached.shape)
                # print("state", state.shape)
                try:
                    # pdb.set_trace()
                    cache[self.uuid] = torch.cat((cached, state_to_cache), 1)
                    if self.ensemble:
                        state = torch.cat((cached.unsqueeze(1).expand(-1,
                                                                      self.layermasks_len,
                                                                      state.shape[1],
                                                                      state.shape[2]).contiguous().view(-1,
                                                                                                        state.shape[1],
                                                                                                        state.shape[2]), state), 1)
                    else:
                        state = cache[self.uuid]
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

        self.label_smoothing = LabelSmoothingLoss(
            config.label_smoothing or 0,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction='none'
        )

        # layermask predictor
        self.layer_mask_predictor = LayerMaskPredictor(config.embedding_size,
                                                       config.num_layers,
                                                       config.layermask_type,
                                                       config.potential_threshold,
                                                       config.shuffle_lmp_configs,
                                                       config.num_configs,
                                                       config.loss_func,
                                                       config.lmp_eval_mode,
                                                       config.layermask_file,
                                                       config.lmp_config_file,
                                                       config.random_config)
        self.layermask_type = config.layermask_type

        # Allow for overriding the encoders and decoders in dervied classes
        self.encoders = type(self).create_encoders(config)
        self.decoders = self.create_decoders(config)

    @classmethod
    def create_encoders(cls, config):
        ''' Create the transformer encoders '''
        kwargs = {'dropout_p': config.dropout_p, 'ensemble': True if config.layermask_type == "ensemble" else False}

        if config.num_enc_layers is None:
            config.num_enc_layers = config.num_layers
        if config.ffn_layer == -1:
            ffn_layer = [1] * config.num_enc_layers
        assert len(ffn_layer) == config.num_enc_layers

        attn_config = {'attn_type': config.attn_type,
                       'attn_position': config.attn_position,
                       'attn_param': config.attn_param,
                       'attn_displacement': config.attn_displacement,
                       'num_layers': config.num_enc_layers,
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
                       'ffn_layer': ffn_layer}
        args = [attn_config, config.num_heads, config.embedding_size, config.hidden_dim]

        encoders = nn.ModuleList([
            TransformerEncoderLayer(*args, layer_i, **kwargs)
            for layer_i in range(config.num_enc_layers)
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
        kwargs = {'dropout_p': config.dropout_p, 'span': config.span, 'ensemble': True if config.layermask_type == "ensemble" else False}

        if config.num_dec_layers is None:
            config.num_dec_layers = config.num_layers
        if config.ffn_layer == -1:
            ffn_layer = [1] * config.num_dec_layers
        assert len(ffn_layer) == config.num_dec_layers

        dec_attn_config = {'attn_type': config.dec_attn_type,
                           'attn_position': config.dec_attn_position,
                           'attn_param': config.dec_attn_param,
                           'attn_displacement': config.dec_attn_displacement,
                           'num_layers': config.num_dec_layers,
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
                           'ffn_layer': ffn_layer}
        enc_dec_attn_config = {'attn_type': config.enc_dec_attn_type,
                               'attn_position': config.enc_dec_attn_position,
                               'attn_param': config.enc_dec_attn_param,
                               'attn_displacement': config.enc_dec_attn_displacement,
                               'num_layers': config.num_dec_layers,
                               'num_heads': config.num_heads,
                               'word_count_ratio': self.dataset.word_count_ratio,
                               'attn_concat': config.enc_dec_attn_concat,
                               'which_attn': 'source',
                               'attn_weights': config.enc_dec_attn_weights,
                               'attn_score': config.enc_dec_attn_score,
                               'attn_bins': config.enc_dec_attn_bins,
                               'enc_dec_attn_layer': 1,
                               'enc_dec_attn_num_heads': config.enc_dec_attn_num_heads,
                               'attn_threshold': config.enc_dec_attn_threshold,
                               'attn_window': config.enc_dec_attn_window,
                               'attn_indexing': config.enc_dec_attn_indexing,
                               'indexing_type': config.indexing_type,
                               'ffn_layer': ffn_layer
                               }
        args = [dec_attn_config, enc_dec_attn_config, config.num_heads, config.embedding_size, config.hidden_dim]
        decoders = nn.ModuleList([
            TransformerDecoderLayer(*args, layer_i,
                                    self.layer_mask_predictor.layermasks.shape[0] if config.layermask_type == "ensemble" else None,
                                    **kwargs)
            for layer_i in range(config.num_dec_layers)
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

        encoded, raw_layermask = self.encode(batch['inputs'])
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
        # pdb.set_trace()

        encoded = {
            'state': word_embedding,
            'mask': inputs.eq(self.padding_idx)
        }

        if raw_layermask is None:
            raw_layermask = self.layer_mask_predictor(encoded['state'], encoded['mask'])

        if self.layermask_type == "ensemble":
            batch_size = word_embedding.shape[0]
            length = encoded['state'].shape[1]
            emb_dim = encoded['state'].shape[2]
            raw_layermask_shape = raw_layermask.shape
            encoded['state'] = encoded['state'].unsqueeze(1).expand(batch_size, raw_layermask_shape[0], length, emb_dim).contiguous().view(-1, length, emb_dim)
            encoded['mask'] = encoded['mask'].unsqueeze(1).expand(batch_size, raw_layermask_shape[0], length).contiguous().view(-1, length)
            ensemble_layermask = raw_layermask.unsqueeze(0).expand(batch_size, raw_layermask_shape[0],
                                                                   raw_layermask_shape[1]).contiguous().view(-1,
                                                                                                raw_layermask_shape[1])

        # pdb.set_trace()

        for i, encoder in enumerate(self.encoders):
            if self.layermask_type == "ensemble":
                encoded = encoder(encoded, i, word_embedding,
                                  gating_weight=ensemble_layermask[:, i])
            elif len(raw_layermask.shape) == 1:
                if raw_layermask[i]:
                    encoded = encoder(encoded, i, word_embedding, gating_weight=raw_layermask[i])
            else:
                encoded = encoder(encoded, i, word_embedding, gating_weight=raw_layermask[:, i])
                
        return encoded, raw_layermask

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

        # pdb.set_trace()

        if self.layermask_type == "ensemble":
            batch_size = word_embedding.shape[0]
            length = decoded['state'].shape[1]
            emb_dim = decoded['state'].shape[2]
            raw_layermask_shape = raw_layermask.shape
            decoded['state'] = decoded['state'].unsqueeze(1).expand(batch_size, raw_layermask_shape[0], length,
                                                                    emb_dim).contiguous().view(-1, length, emb_dim)
            decoded['mask'] = decoded['mask'].unsqueeze(1).expand(batch_size, raw_layermask_shape[0], length).contiguous().view(-1,
                                                                                                                   length)
            # pdb.set_trace()
            ensemble_layermask = raw_layermask.unsqueeze(0).expand(batch_size, raw_layermask_shape[0],
                                                                   raw_layermask_shape[1]).contiguous().view(-1,
                                                                                                raw_layermask_shape[1])

        # if len(raw_layermask) != encoded['state'].shape[0]: # for debugging beam_search
        #     pdb.set_trace()

        # assert raw_layermask is not None
        # pdb.set_trace()

        for i, decoder in enumerate(decoders):
            if self.layermask_type == "ensemble":
                decoded = decoder(decoded, encoded, i, word_embedding, gating_weight=ensemble_layermask[:, len(decoders) + i])
            elif len(raw_layermask.shape) == 1:
                if raw_layermask[len(decoders) + i]:
                    decoded = decoder(decoded, encoded, i, word_embedding,
                                      gating_weight=raw_layermask[len(decoders) + i])
            else:
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
        # used by action:itertrain
        if lmp_type not in ["random", "noskip", "itertrain"]:
            raise ValueError

        self.layer_mask_predictor.lmp_type = lmp_type

