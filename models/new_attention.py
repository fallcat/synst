'''
A module which implements various attention mechanisms
'''
import math
import torch
import time
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.attention import MultiHeadedAttention

from utils import same_tensor, pad_unsorted_sequence


class NewAttention(nn.Module):
    ''' Implement a hard-coded attention module '''
    ATTN_TYPES = ['normal', 'uniform', 'no', 'learned']
    ATTN_POSITIONS = ['center', 'left', 'right', 'first', 'last']

    def __init__(self, attn_config, embed_dim, num_heads=1):
        ''' Initialize the attention module '''
        super(NewAttention, self).__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = self.projection_dim ** -0.5
        self.attn_type = attn_config['attn_type']
        self.attn_position = attn_config['attn_position']
        self.attn_param = attn_config['attn_param']
        self.attn_displacement = attn_config['attn_displacement']
        self.num_layers = attn_config['num_layers']
        self.word_count_ratio = attn_config['word_count_ratio'] if 'word_count_ratio' in attn_config else 1
        self.attn_concat = attn_config['attn_concat'] if 'attn_concat' in attn_config else 0
        if self.attn_concat in [1, 2]:
            self.attn_concat_weights = nn.Parameter(torch.Tensor(embed_dim, 2 * embed_dim))
        elif self.attn_concat == 3:
            self.attn_concat_weights = nn.Parameter(torch.Tensor(embed_dim, 3 * embed_dim))
        else:
            self.attn_concat_weights = None
        self.which_attn = attn_config['which_attn']
        self.attn_score = attn_config['attn_score']
        if self.attn_score:
            self.attn_score_project_in_weights = nn.Parameter(torch.Tensor(self.projection_dim, embed_dim))
            self.attn_score_project_out_weights = nn.Parameter(torch.Tensor(embed_dim, self.projection_dim))
        self.attn_bins = attn_config['attn_bins']

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.attn_linear_transform = attn_config['attn_weights']
        self.input_weights = None
        if self.attn_linear_transform:
            if 'learned' in self.attn_type or 'learned' == self.attn_type:
                if self.attn_linear_transform == 1:
                    self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
                elif self.attn_linear_transform == 2:
                    self.input_weights = nn.Parameter(torch.Tensor(2 * embed_dim, embed_dim))
            else:
                self.input_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()
        self.attn_configs = list(self.load_attn_configs())

        self.attn_weights = {}

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        if self.input_weights is not None:
            nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)
        if self.attn_concat_weights is not None:
            nn.init.xavier_uniform_(self.attn_concat_weights, gain)
        if self.attn_score:
            nn.init.xavier_uniform_(self.attn_score_project_in_weights, gain)
            nn.init.xavier_uniform_(self.attn_score_project_out_weights, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def project_learned(self, inputs, learned_idx):
        batch_size = int(inputs.shape[0] / self.num_heads)
        return inputs.view(batch_size,
                           self.num_heads,
                           -1,
                           self.projection_dim)[:, learned_idx].contiguous()\
            .view(batch_size * len(learned_idx),
                  -1,
                  self.projection_dim)

    def load_attn_configs(self):
        for layer_i in range(self.num_layers):
            attn_configs = []

            attn_configs_names = ['attn_type', 'attn_position', 'attn_param', 'attn_displacement']

            for i, attn_config_i in enumerate(
                    [self.attn_type, self.attn_position, self.attn_param, self.attn_displacement]):
                len_attn_config_i = len(attn_config_i)
                if type(attn_config_i) is list:
                    if len_attn_config_i == 1:
                        attn_configs.append(attn_config_i[0])
                    elif len_attn_config_i == self.num_heads:
                        if len(set(attn_config_i)) == 1:
                            attn_configs.append(attn_config_i[0])
                        else:
                            attn_configs.append(attn_config_i)
                    elif len_attn_config_i == self.num_layers:
                        attn_configs.append(attn_config_i[layer_i])
                    elif len_attn_config_i == self.num_heads * self.num_layers:
                        if len(set(attn_config_i[layer_i * self.num_heads:(layer_i + 1) * self.num_heads])) == 1:
                            attn_configs.append(attn_config_i[layer_i * self.num_heads])
                        else:
                            attn_configs.append(attn_config_i[layer_i * self.num_heads:(layer_i + 1) * self.num_heads])
                    elif len_attn_config_i < self.num_heads and self.num_heads % len_attn_config_i == 0:
                        attn_configs.append(attn_config_i * (self.num_heads // len_attn_config_i))
                    elif len_attn_config_i % self.num_layers == 0 and \
                            len_attn_config_i < self.num_heads * self.num_layers and \
                            self.num_heads % (len_attn_config_i // self.num_layers) == 0:
                        num_each_head = len_attn_config_i // self.num_layers
                        repeat_each_head = self.num_heads // num_each_head
                        attn_configs.append(attn_config_i[layer_i * num_each_head:(layer_i + 1) * num_each_head] *
                                            repeat_each_head)
                    else:
                        raise Exception("The number of {} is {}, but it has to be either number of heads {}, "
                                        "number of layers {}, or the product of them {}.".format(attn_configs_names[i],
                                                                                                 len(attn_config_i),
                                                                                                 self.num_heads,
                                                                                                 self.num_layers,
                                                                                                 self.num_heads * self.num_layers))
                else:
                    attn_configs.append(attn_config_i)
            yield attn_configs

    def attention(self, values, keys, queries, key_mask=None, mask=None, layer_i=0, decoder_position=-1, input_lens=None):
        ''' Scaled dot product attention with optional masks '''

        # print("values", values.shape)
        # print("keys", keys.shape)
        # print("queries", queries.shape)
        # print("attn_type", self.attn_type)
        # print("attn_position", self.attn_position)
        # print("input weights", self.input_weights)
        # print("decoder_position", decoder_position)
        # print("target_lens", target_lens)
        queries_shape = queries.shape
        values_shape = values.shape
        # print("queries_shape", queries_shape)
        # print("values_shape", values_shape)
        # print("self.word_count_ratio", self.word_count_ratio)


        # By this point the values, keys, and queries all have B * H as their first dimension
        batch_size = queries_shape[0] // self.num_heads

        attn_type, attn_position, attn_param, attn_displacement = self.attn_configs[layer_i]

        if attn_type == 'learned':
            time1 = time.time()
            logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))
            if mask is not None:
                logits += mask

            if key_mask is not None:
                logits_shape = logits.shape
                batch_size = logits_shape[0] // self.num_heads
                logits = logits.view(batch_size, self.num_heads, logits_shape[1], logits_shape[2])
                logits.masked_fill_(key_mask[:, None, None], float('-inf'))
                logits = logits.view(logits_shape)

            attn_weights = F.softmax(logits, dim=-1)

            attended = torch.bmm(attn_weights, values)

            batch_size = queries_shape[0] // self.num_heads

            # print("time for learned:", time.time() - time1)

            return attended.view(
                        batch_size,
                        self.num_heads,
                        -1,
                        self.projection_dim
                    ).transpose(2, 1).contiguous().view(
                        batch_size,
                        -1,
                        self.num_heads * self.projection_dim
                    )

        elif 'learned' in attn_type:
            time1 = time.time()
            learned_idx = np.where(np.array(attn_type) == 'learned')[0]
            len_learned_idex = len(learned_idx)
            queries_ = self.project_learned(queries, learned_idx)
            keys_ = self.project_learned(keys, learned_idx)
            values_ = self.project_learned(values, learned_idx)

            logits_ = self.scale * torch.bmm(queries_, keys_.transpose(2, 1))
            logits_shape_ = logits_.shape
            if mask is not None:
                logits_ += mask

            if key_mask is not None:
                batch_size = logits_shape_[0] // len_learned_idex
                logits_ = logits_.view(batch_size, len_learned_idex, logits_shape_[1], logits_shape_[2])
                logits_.masked_fill_(key_mask[:, None, None], float('-inf'))
                logits_ = logits_.view(logits_shape_)
            logits_ = F.softmax(logits_, dim=-1).view(batch_size,
                                                      len(learned_idx),
                                                      logits_shape_[-2],
                                                      logits_shape_[-1])

            learned_count = 0
            # print("time for partially learned:", time.time() - time1)

        if not {'last', 'bin'}.isdisjoint(attn_position) or attn_position in ['last', 'bin']:
            time2 = time.time()
            if input_lens is not None:
                last_indices = (input_lens - 1).cpu().view(-1)
            elif key_mask is not None:
                # print("key_mask is not none")
                key_mask_shape = key_mask.shape
                # last_indices = torch.tensor([key_mask_shape[1] - a[::-1].index(0)
                #                              for a in key_mask.cpu().numpy().tolist()], dtype=torch.float32).view(-1, 1)
                last_indices = ((key_mask == 0).sum(dim=1) - 1).cpu().view(-1)   # .tolist()
                # print("last_indices", last_indices)
            else:
                # print("key_mask is none")
                last_indices = torch.tensor([values_shape[1] - 1] * queries_shape[0], dtype=torch.float32).view(-1)
                # print("last_indices", last_indices)
            # print("calculate last_indices", time.time() - time2)

        if type(attn_type) is not list and type(attn_position) is not list and type(attn_param) is not list and type(attn_displacement) is not list:
            time3 = time.time()
            need_recompute = False
            if attn_type not in self.attn_weights:
                self.attn_weights[attn_type] = {}
            if attn_position not in self.attn_weights[attn_type]:
                self.attn_weights[attn_type][attn_position] = {}
            if attn_position == 'center':
                if attn_param not in self.attn_weights[attn_type][attn_position] \
                        or (queries_shape[1] > self.attn_weights[attn_type][attn_position][attn_param].shape[0]
                            or values_shape[1] > self.attn_weights[attn_type][attn_position][attn_param].shape[1]):
                    need_recompute = True
            elif attn_position == 'first':
                if attn_param not in self.attn_weights[attn_type][attn_position] \
                        or values_shape[1] > self.attn_weights[attn_type][attn_position][attn_param].shape[1]:
                    need_recompute = True
            else:
                if attn_position in ['left', 'right']:
                    if attn_param not in self.attn_weights[attn_type][attn_position]:
                        self.attn_weights[attn_type][attn_position][attn_param] = {}
                        need_recompute = True
                    if attn_displacement not in self.attn_weights[attn_type][attn_position][attn_param] \
                            or (queries_shape[1] > self.attn_weights[attn_type][attn_position][attn_param][attn_displacement].shape[0]
                                or values_shape[1] > self.attn_weights[attn_type][attn_position][attn_param][attn_displacement].shape[1]):
                        need_recompute = True
                else:  # attn_position in ['last', 'bin']
                    # last_indices_set = set(last_indices)
                    max_last_index = last_indices[0].cpu().item()
                    if attn_position == 'last':
                        if attn_param not in self.attn_weights[attn_type][attn_position] \
                                or max_last_index + 1 > self.attn_weights[attn_type][attn_position][attn_param].shape[0]:
                            need_recompute = True
                    else:
                        if attn_param not in self.attn_weights[attn_type][attn_position]:
                            self.attn_weights[attn_type][attn_position][attn_param] = {}
                            need_recompute = True
                        elif max_last_index + 1 > self.attn_weights[attn_type][attn_position][attn_param][attn_displacement].shape[0]:
                            need_recompute = True

            # print("conditions", time.time() - time3)
            time4 = time.time()

            if need_recompute:
                indices_v = torch.arange(values_shape[1]).view(1, -1).to(dtype=torch.float32)

                if attn_position not in ['last', 'bin']:
                    # if attn_position == 'bin':
                    #     bin_center = -0.5 + values_shape[1] * (attn_displacement - 0.5) / self.attn_bins
                    #     indices_q = torch.full((queries_shape[1], 1),
                    #                            bin_center).to(dtype=torch.float32)
                    if attn_position == 'first':
                        indices_q = torch.tensor(0.0, dtype=torch.float32)# torch.full((queries_shape[1], 1), 0).to(dtype=torch.float32)
                    elif decoder_position == -1:
                        indices_q = torch.arange(queries_shape[1]
                                                 ).view(-1, 1).to(dtype=torch.float32) * self.word_count_ratio
                    else:
                        indices_q = torch.full((queries_shape[1], 1),
                                               decoder_position * self.word_count_ratio).to(dtype=torch.float32)
                    # print("attn_position", attn_position)
                    if attn_position == 'left':
                        indices_q = indices_q - attn_displacement
                    elif attn_position == 'right':
                        indices_q = indices_q + attn_displacement

                    distance_diff = indices_v - indices_q

                # If the attention is looking at the last indices, need to take masks into consideration
                else:
                    # new_last_indices_list = list(new_last_indices_set)
                    # indices_q = torch.tensor(new_last_indices_list).view(-1, 1).to(dtype=torch.float32)
                    indices_q = torch.arange(max_last_index + 1).view(-1, 1).to(dtype=torch.float32)
                    old_indices_q = indices_q
                    if attn_position == 'bin':
                        ratio = (attn_displacement - 0.5) / self.attn_bins
                        indices_q = -0.5 + indices_q * ratio
                    distance_diff = (indices_v - indices_q)

                # print("diff", time.time() - time4)
                time5 = time.time()

                if attn_type == 'normal':
                    std = attn_param
                    logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / std) ** 2))
                else:
                    if attn_param < 0 and attn_position == 'bin':
                        attn_param_curr = (0.5 * old_indices_q / self.attn_bins).view(-1, 1)
                    else:
                        attn_param_curr = attn_param
                    # print("distance_diff", distance_diff.shape)
                    # print("attn_param_curr", attn_param_curr.shape)
                    distance_diff = torch.abs(distance_diff)
                    distance_diff[distance_diff <= attn_param_curr] = 0
                    distance_diff[distance_diff > attn_param_curr] = 1
                    logits = 1 - distance_diff
                    logits_sum = torch.sum(logits, dim=-1, keepdim=True)
                    logits_sum[logits_sum == 0] = 1
                    logits = logits / logits_sum

                # print("normal or uniform", time.time() - time5)
                time6 = time.time()

                if attn_position in ['center', 'first']:
                    self.attn_weights[attn_type][attn_position][attn_param] = logits
                elif attn_position in ['left', 'right']:
                    self.attn_weights[attn_type][attn_position][attn_param][attn_displacement] = logits
                elif attn_position == 'last':
                    self.attn_weights[attn_type][attn_position][attn_param] = logits
                    # self.attn_weights[attn_type][attn_position][attn_param].update({new_last_indices_list[i]:row[0][:, :new_last_indices_list[i] + 1] for i, row in enumerate(logits)})
                else:
                    self.attn_weights[attn_type][attn_position][attn_param][attn_displacement] = logits
                    # self.attn_weights[attn_type][attn_position][attn_param][attn_displacement].update(
                    #     {new_last_indices_list[i]: row[0][:, :new_last_indices_list[i] + 1] for i, row in enumerate(logits)})

                # print("store", time.time() - time6)

            time7 = time.time()

            if attn_position == 'center':
                logits = self.attn_weights[attn_type][attn_position][attn_param][:queries_shape[1], :values_shape[1]].unsqueeze(0).unsqueeze(0)
            elif attn_position == 'first':
                logits = self.attn_weights[attn_type][attn_position][attn_param][:, :values_shape[1]].unsqueeze(0).unsqueeze(0)
            elif attn_position in ['left', 'right']:
                logits = self.attn_weights[attn_type][attn_position][attn_param][attn_displacement][:queries_shape[1], :values_shape[1]].unsqueeze(0).unsqueeze(0)
            elif attn_position == 'last':
                logits = torch.index_select(self.attn_weights[attn_type][attn_position][attn_param], 0, last_indices)[:, :values_shape[1]].unsqueeze(1).unsqueeze(1)
            else:
                time71 = time.time()
                logits = torch.index_select(self.attn_weights[attn_type][attn_position][attn_param][attn_displacement], 0, last_indices)[:, :values_shape[1]].unsqueeze(1).unsqueeze(1)


            logits = logits.expand(batch_size, self.num_heads, queries_shape[1], values_shape[1])\
                .contiguous().view(-1,
                                   queries_shape[1],
                                   values_shape[1])
            if self.which_attn == 'source':
                print("retrieve", time.time() - time7)
                print("need compute", need_recompute, "time", time.time() - time3)

            attn_weights = logits.type_as(values)
            # print("attn_weights 1", attn_weights)

        # If one of the attention parameters is list (different in different heads), then make all of them lists
        else:
            time3 = time.time()
            attn_config = []
            for attn_config_i in [attn_type, attn_position, attn_param, attn_displacement]:
                if type(attn_config_i) is not list:
                    attn_config.append([attn_config_i] * self.num_heads)
                else:
                    attn_config.append(attn_config_i)

            attn_type, attn_position, attn_param, attn_displacement = attn_config

            logits_list = []

            for i in range(self.num_heads):
                time4 = time.time()
                if attn_type[i] == 'learned':
                    logits = logits_[:, learned_count]
                    learned_count += 1
                else:
                    need_recompute = False
                    if attn_type[i] not in self.attn_weights:
                        self.attn_weights[attn_type[i]] = {}
                    if attn_position[i] not in self.attn_weights[attn_type[i]]:
                        self.attn_weights[attn_type[i]][attn_position[i]] = {}
                    if attn_position[i] == 'center':
                        if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]] \
                                or (queries_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[0]
                                    or values_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[
                                        1]):
                            need_recompute = True
                    elif attn_position[i] == 'first':
                        if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]] \
                                or values_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[1]:
                            need_recompute = True
                    else:
                        if attn_position[i] in ['left', 'right']:
                            if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]]:
                                self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = {}
                                need_recompute = True
                            if attn_displacement[i] not in self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] \
                                    or (queries_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][
                                attn_displacement[i]].shape[0]
                                        or values_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][
                                            attn_displacement[i]].shape[1]):
                                need_recompute = True
                        else:  # attn_position[i] in ['last', 'bin']
                            # last_indices_set = set(last_indices)
                            max_last_index = last_indices[0].cpu().item()
                            if attn_position[i] == 'last':
                                if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]] \
                                        or max_last_index + 1 > \
                                        self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[0]:
                                    need_recompute = True
                            else:
                                if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]]:
                                    self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = {}
                                    need_recompute = True
                                elif max_last_index + 1 > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][
                                    attn_displacement[i]].shape[0]:
                                    need_recompute = True

                    # print("conditions", time.time() - time3)
                    time4 = time.time()

                    if need_recompute:
                        indices_v = torch.arange(values_shape[1]).view(1, -1).to(dtype=torch.float32)

                        if attn_position[i] not in ['last', 'bin']:
                            # if attn_position[i] == 'bin':
                            #     bin_center = -0.5 + values_shape[1] * (attn_displacement[i] - 0.5) / self.attn_bins
                            #     indices_q = torch.full((queries_shape[1], 1),
                            #                            bin_center).to(dtype=torch.float32)
                            if attn_position[i] == 'first':
                                indices_q = torch.tensor(0.0,
                                                         dtype=torch.float32)  # torch.full((queries_shape[1], 1), 0).to(dtype=torch.float32)
                            elif decoder_position == -1:
                                indices_q = torch.arange(queries_shape[1]
                                                         ).view(-1, 1).to(dtype=torch.float32) * self.word_count_ratio
                            else:
                                indices_q = torch.full((queries_shape[1], 1),
                                                       decoder_position * self.word_count_ratio).to(dtype=torch.float32)
                            # print("attn_position[i]", attn_position[i])
                            if attn_position[i] == 'left':
                                indices_q = indices_q - attn_displacement[i]
                            elif attn_position[i] == 'right':
                                indices_q = indices_q + attn_displacement[i]

                            distance_diff = indices_v - indices_q

                        # If the attention is looking at the last indices, need to take masks into consideration
                        else:
                            # new_last_indices_list = list(new_last_indices_set)
                            # indices_q = torch.tensor(new_last_indices_list).view(-1, 1).to(dtype=torch.float32)
                            indices_q = torch.arange(max_last_index + 1).view(-1, 1).to(dtype=torch.float32)
                            old_indices_q = indices_q
                            if attn_position[i] == 'bin':
                                ratio = (attn_displacement[i] - 0.5) / self.attn_bins
                                indices_q = -0.5 + indices_q * ratio
                            distance_diff = (indices_v - indices_q)

                        # print("diff", time.time() - time4)
                        time5 = time.time()

                        if attn_type[i] == 'normal':
                            std = attn_param[i]
                            logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(
                                - 1 / 2 * (distance_diff / std) ** 2))
                        else:
                            if attn_param[i] < 0 and attn_position[i] == 'bin':
                                attn_param_curr = (0.5 * old_indices_q / self.attn_bins).view(-1, 1)
                            else:
                                attn_param_curr = attn_param[i]
                            # print("distance_diff", distance_diff.shape)
                            # print("attn_param_curr", attn_param_curr.shape)
                            distance_diff = torch.abs(distance_diff)
                            distance_diff[distance_diff <= attn_param_curr] = 0
                            distance_diff[distance_diff > attn_param_curr] = 1
                            logits = 1 - distance_diff
                            logits_sum = torch.sum(logits, dim=-1, keepdim=True)
                            logits_sum[logits_sum == 0] = 1
                            logits = logits / logits_sum

                        # print("normal or uniform", time.time() - time5)
                        time6 = time.time()

                        if attn_position[i] in ['center', 'first']:
                            self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = logits
                        elif attn_position[i] in ['left', 'right']:
                            self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]] = logits
                        elif attn_position[i] == 'last':
                            self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = logits
                            # self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].update({new_last_indices_list[i]:row[0][:, :new_last_indices_list[i] + 1] for i, row in enumerate(logits)})
                        else:
                            self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]] = logits
                            # self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]].update(
                            #     {new_last_indices_list[i]: row[0][:, :new_last_indices_list[i] + 1] for i, row in enumerate(logits)})

                        # print("store", time.time() - time6)

                    time7 = time.time()

                    if attn_position[i] == 'center':
                        logits = self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][:queries_shape[1],
                                 :values_shape[1]].unsqueeze(0).unsqueeze(0)
                    elif attn_position[i] == 'first':
                        logits = self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][:, :values_shape[1]].unsqueeze(
                            0).unsqueeze(0)
                    elif attn_position[i] in ['left', 'right']:
                        logits = self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]][
                                 :queries_shape[1], :values_shape[1]].unsqueeze(0).unsqueeze(0)
                    elif attn_position[i] == 'last':
                        logits = torch.index_select(self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]], 0,
                                                    last_indices)[:, :values_shape[1]].unsqueeze(1).unsqueeze(1)
                    else:
                        time71 = time.time()
                        logits = torch.index_select(
                            self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]], 0,
                            last_indices)[:, :values_shape[1]].unsqueeze(1).unsqueeze(1)

                    logits = logits.expand(batch_size, 1, queries_shape[1], values_shape[1])
                    # print("logits", logits.shape)
                    print("new time in loop", time.time() - time4)
                    time5 = time.time()


                    # If the attention weight matrix is not stored, need to create new.
                    # At inference time, always create new for decoder attentions.
                    # If attention position is last or middle, always recalculate because the stored is wrong.

                    indices_v = torch.arange(values_shape[1]).view(1, -1).to(dtype=torch.float32)
                    # print("time3", time.time() - time3)
                    time4 = time.time()

                    if attn_position[i] not in ['last', 'bin']:
                        # if attn_position[i] == 'bin':
                        #     bin_center = -0.5 + values_shape[1] * (attn_displacement[i] - 0.5) / self.attn_bins
                        #     indices_q = torch.full((queries_shape[1], 1),
                        #                            bin_center).to(dtype=torch.float32)
                        if attn_position[i] == 'first':
                            indices_q = torch.full((queries_shape[1], 1),
                                                   0).to(dtype=torch.float32)
                        elif decoder_position == -1:
                            indices_q = torch.arange(queries_shape[1]
                                                     ).view(-1, 1).to(dtype=torch.float32) * self.word_count_ratio

                        else:
                            indices_q = torch.full((queries_shape[1], 1),
                                                   decoder_position * self.word_count_ratio).to(dtype=torch.float32)

                        if attn_position[i] == 'left':
                            indices_q = indices_q - attn_displacement[i]
                        elif attn_position[i] == 'right':
                            indices_q = indices_q + attn_displacement[i]

                        distance_diff = indices_v - indices_q

                        distance_diff = distance_diff.expand(batch_size,
                                                             distance_diff.shape[0],
                                                             distance_diff.shape[1])

                    # If the attention is looking at the last indices, need to take masks into consideration
                    else:
                        last_indices = last_indices.view(-1, 1)
                        indices_q = last_indices
                        if attn_position[i] == 'bin':
                            ratio = (attn_displacement[i] - 0.5) / self.attn_bins
                            indices_q = -0.5 + indices_q * ratio
                        distance_diff = (indices_v - indices_q).unsqueeze(1)
                        distance_diff = distance_diff.expand(batch_size, queries_shape[1],
                                                             values_shape[1]).contiguous()

                    # print("distance_diff time", time.time() - time4)
                    time5 = time.time()

                    if attn_type[i] == 'normal':
                        std = attn_param[i]

                        logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / std) ** 2))
                        # print("time logits normal", time.time() - time5)
                    else:
                        if attn_param[i] < 0 and attn_position[i] == 'bin':
                            attn_param_curr = (0.5 * last_indices / self.attn_bins).unsqueeze(-1)
                        else:
                            attn_param_curr = attn_param[i]
                        distance_diff = torch.abs(distance_diff)
                        distance_diff[distance_diff <= attn_param_curr] = 0
                        distance_diff[distance_diff > attn_param_curr] = 1
                        logits = 1 - distance_diff
                        logits_sum = torch.sum(logits, dim=-1, keepdim=True)
                        logits_sum[logits_sum == 0] = 1
                        logits = logits / logits_sum
                        # print("time logits uniform", time.time() - time5)
                        # logits = F.softmax(logits, dim=-1)
                    self.attn_weights[attn_type[i]][attn_position[i]] = logits[0]

                        # print("retrieving weights", time.time() - time3)
                    logits = logits.type_as(values)
                    print("old time in loop", time.time() - time5)
                logits_list.append(logits)

                if self.which_attn == 'source':
                    print("time in loop", time.time() - time3)
            if self.which_attn == 'source':
                print("final time", time.time() - time3)
            attn_weights = torch.stack(logits_list, dim=1)
            print("attn_weights", attn_weights.shape)
            attn_weights = attn_weights.view(values_shape[0],
                                             queries_shape[1],
                                             values_shape[1])
        if mask is not None:
            new_mask = mask.clone()
            new_mask[new_mask == 0] = 1
            new_mask[new_mask == float('-inf')] = 0
            attn_weights = attn_weights.clone() * new_mask
        if key_mask is not None:
            attn_weights_shape = attn_weights.shape
            batch_size = attn_weights_shape[0] // self.num_heads
            attn_weights = attn_weights.view(batch_size, self.num_heads, attn_weights_shape[1], attn_weights_shape[2])
            attn_weights.masked_fill_(key_mask[:, None, None], float(0))
            attn_weights = attn_weights.view(attn_weights_shape)
        attended = torch.bmm(attn_weights,
                             values)

        # torch.set_printoptions(profile='full')
        # print("values", values)
        # print("values shape", values.shape)
        # torch.set_printoptions(profile="full")
        # if self.which_attn == 'source':
        #     print("attn_weights", attn_weights)
        #     print("attn_weights shape", attn_weights.shape)
        # print("attended", attended)
        # print("attended shape", attended.shape)

        return attended.view(
            batch_size,
            self.num_heads,
            -1,
            self.projection_dim
        ).transpose(2, 1).contiguous().view(
            batch_size,
            -1,
            self.num_heads * self.projection_dim
        )

    def forward(self, values, keys, queries, # pylint:disable=arguments-differ
                key_mask=None, attention_mask=None, num_queries=0, layer_i=0, decoder_position=-1, input_lens=None,
                original_targets=None, word_embedding=None):
        ''' Forward pass of the attention '''
        batch_size = values.shape[0]
        # print("key_mask", key_mask)

        if 'learned' in self.attn_type or 'learned' == self.attn_type:
            if self.attn_linear_transform == 1:
                if same_tensor(values, keys, queries):
                    values, keys, queries = self.project(values, chunks=3)
                elif same_tensor(values, keys):
                    values, keys = self.project(values, chunks=2)
                    queries, = self.project(queries, 2)
                else:
                    values, = self.project(values, 0)
                    keys, = self.project(keys, 1)
                    queries, = self.project(queries, 2)
            elif self.attn_linear_transform == 2:
                if same_tensor(keys, queries):
                    keys, queries = self.project(queries, chunks=2)
                else:
                    keys, = self.project(keys, 0)
                    queries, = self.project(queries, 1)
                values = values.view(batch_size,
                                     -1,
                                     self.num_heads,
                                     self.projection_dim
                                     ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            else:
                inputs = []
                for inp in [values, keys, queries]:
                    inputs.append(inp.view(batch_size,
                                           -1,
                                           self.num_heads,
                                           self.projection_dim
                                           ).transpose(2, 1).contiguous().view(
                        batch_size * self.num_heads,
                        -1,
                        self.projection_dim
                    ))
                values, keys, queries = inputs
        else:
            if self.attn_linear_transform:
                values = F.linear(values, self.input_weights)
            values = values.view(
                batch_size,
                -1,
                self.num_heads,
                self.projection_dim
            ).transpose(2, 1).contiguous().view(
                batch_size * self.num_heads,
                -1,
                self.projection_dim
            )

            queries = queries.view(
                batch_size,
                -1,
                self.num_heads,
                self.projection_dim
            ).transpose(2, 1).contiguous().view(
                batch_size * self.num_heads,
                -1,
                self.projection_dim
            )
        # pylint:enable=unbalanced-tuple-unpacking

        if num_queries:
            queries = queries[:, -num_queries:]

        attended = self.attention(values, keys, queries, key_mask, attention_mask, layer_i, decoder_position,
                                  input_lens)

        queries = queries.view(
            batch_size,
            self.num_heads,
            -1,
            self.projection_dim
        ).transpose(2, 1).contiguous().view(
            batch_size,
            -1,
            self.num_heads * self.projection_dim
        )

        if self.attn_score:
            projected_queries = F.linear(queries, self.attn_score_project_in_weights).view(-1,
                                                                                           1,
                                                                                           self.projection_dim)
            attended_shape = attended.shape
            attended = attended.view(-1,
                                     self.num_heads,
                                     self.projection_dim)
            scores = torch.bmm(projected_queries, attended.transpose(1, 2)).softmax(dim=-1)
            attended = F.linear(torch.bmm(scores, attended).squeeze(1),
                                self.attn_score_project_out_weights).view(attended_shape)

        if 'learned' not in self.attn_type and 'learned' != self.attn_type and self.attn_concat_weights is not None:
            if self.attn_concat == 1:
                attended = F.linear(torch.cat((attended, queries), dim=-1), self.attn_concat_weights)
            elif self.attn_concat == 2:
                attended = F.linear(torch.cat((attended, word_embedding), dim=-1), self.attn_concat_weights)
            else:
                attended = F.linear(torch.cat((attended, queries, word_embedding), dim=-1), self.attn_concat_weights)

        return self.output_projection(attended)
