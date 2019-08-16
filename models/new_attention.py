'''
A module which implements various attention mechanisms
'''
import math
import torch
import time
import numpy as np
from torch import nn
from torch.nn import functional as F

from utils import same_tensor


class NewAttention(nn.Module):
    ''' Implement a hard-coded attention module '''
    ATTN_TYPES = ['normal', 'uniform', 'whole', 'no', 'learned']
    ATTN_POSITIONS = ['center', 'left', 'right', 'first', 'last', 'middle']

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
        # self.max_prob = attn_config['max_prob']
        # self.window_size = attn_config['window_size']

        # Combine projections for multiple heads into a single linear layer for efficiency
        # self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if 'learned' in self.attn_type:
            self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.input_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

        self.attn_weights = {}

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

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

    def attention(self, values, keys, queries, key_mask=None, mask=None, layer_i=0):
        ''' Scaled dot product attention with optional masks '''

        # By this point the values, keys, and queries all have B * H as their first dimension
        batch_size = queries.shape[0] // self.num_heads

        attn_configs = []

        for i, attn_config_i in enumerate([self.attn_type, self.attn_position, self.attn_param, self.attn_displacement]):
            if type(attn_config_i) is list:
                if len(attn_config_i) == 1:
                    attn_configs.append(attn_config_i[0])
                elif len(attn_config_i) == self.num_heads:
                    if len(set(attn_config_i)) == 1:
                        attn_configs.append(attn_config_i[0])
                    attn_configs.append(attn_config_i)
                elif len(attn_config_i) == self.num_layers:
                    attn_configs.append(attn_config_i[layer_i])
                else:
                    if len(set(attn_config_i[layer_i * self.num_heads:(layer_i + 1) * self.num_heads])) == 1:
                        attn_configs.append(attn_config_i[layer_i * self.num_heads])
                    else:
                        attn_configs.append(attn_config_i[layer_i * self.num_heads:(layer_i + 1) * self.num_heads])
            else:
                attn_configs.append(attn_config_i)
        attn_type, attn_position, attn_param, attn_displacement = attn_configs

        if attn_type == 'learned':
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

            batch_size = queries.shape[0] // self.num_heads

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
            learned_idx = np.where(np.array(attn_type) == 'learned')[0]
            len_learned_idex = len(learned_idx)
            queries_ = self.project_learned(queries, learned_idx)
            keys_ = self.project_learned(keys, learned_idx)
            print("keys", keys.shape)
            print("keys_", keys_.shape)
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

        if type(attn_type) is not list and type(attn_position) is not list:
            if attn_type == 'whole':
                logits = torch.full((queries.shape[1], values.shape[1]), 1 / values.shape[1]).to(dtype=torch.float32)
            else:
                if attn_type not in self.attn_weights:
                    self.attn_weights[attn_type] = {}
                if attn_position not in self.attn_weights[attn_type] \
                        or (queries.shape[1] > self.attn_weights[attn_type][attn_position].shape[0]
                            or values.shape[1] > self.attn_weights[attn_type][attn_position].shape[1]):
                    indices_q = torch.arange(queries.shape[1]).view(-1, 1).to(dtype=torch.float32)
                    indices_v = torch.arange(values.shape[1]).view(1, -1).to(dtype=torch.float32)

                    if attn_position == 'left':
                        indices_q = indices_q - attn_displacement
                    elif attn_position == 'right':
                        indices_q = indices_q + attn_displacement
                    elif attn_position == 'first':
                        indices_q[:] = 0
                    elif attn_position == 'last':
                        indices_q[:] = indices_q.size()[0] - 1
                    elif attn_position == 'middle':
                        indices_q[:] = (indices_q.size()[0] + 1) / 2 - 1

                    distance_diff = indices_v - indices_q
                    # print("distance_diff", distance_diff)

                    if attn_type == 'normal':
                        # std = 1 / (attn_param * math.sqrt(2 * math.pi))
                        std = attn_param

                        logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / std) ** 2))
                    else:
                        distance_diff = torch.abs(distance_diff)
                        distance_diff[distance_diff <= attn_param] = 0
                        distance_diff[distance_diff > attn_param] = 1
                        logits = 1 - distance_diff
                        logits = logits / torch.sum(logits, dim=-1, keepdim=True)
                        # logits = F.softmax(logits, dim=-1)
                    self.attn_weights[attn_type][attn_position] = logits
                else:
                    logits = self.attn_weights[attn_type][attn_position][:queries.shape[1], :values.shape[1]]
            # print("logits", logits)
            attn_weights = logits.type_as(values)
            attended = torch.bmm(attn_weights.expand(values.shape[0], attn_weights.shape[0], attn_weights.shape[1]),
                                 values)

        else:
            if type(attn_type) is not list:
                # print("attn_type not list")
                attn_type = [attn_type] * self.num_heads
            if type(attn_position) is not list:
                # print("attn_position not list")
                attn_position = [attn_position] * self.num_heads
            if type(attn_param) is not list:
                # print("attn_param not list")
                attn_param = [attn_param] * self.num_heads
            if type(attn_displacement) is not list:
                # print("attn_param not list")
                attn_displacement = [attn_displacement] * self.num_heads
            logits_list = []
            for i in range(self.num_heads):
                if attn_type[i] == 'whole':
                    logits = torch.full((queries.shape[1], values.shape[1]), 1 / values.shape[1]).to(
                        dtype=torch.float32)\
                        .unsqueeze(0)\
                        .expand(int(values.shape[0] / self.num_heads),
                                queries.shape[1],
                                values.shape[1])
                elif attn_type[i] == 'learned':
                    logits = logits_[:, learned_count]
                    learned_count += 1
                else:
                    if attn_type[i] not in self.attn_weights:
                        self.attn_weights[attn_type[i]] = {}
                    if attn_position[i] not in self.attn_weights[attn_type[i]] \
                            or (queries.shape[1] > self.attn_weights[attn_type[i]][attn_position[i]].shape[0]
                                or values.shape[1] > self.attn_weights[attn_type[i]][attn_position[i]].shape[1]):
                        indices_q = torch.arange(queries.shape[1]).view(-1, 1).to(dtype=torch.float32)
                        indices_v = torch.arange(values.shape[1]).view(1, -1).to(dtype=torch.float32)

                        if attn_position[i] == 'left':
                            indices_q = indices_q - attn_displacement[i]
                        elif attn_position[i] == 'right':
                            indices_q = indices_q + attn_displacement[i]
                        elif attn_position[i] == 'first':
                            indices_q[:] = 0
                        elif attn_position[i] == 'last':
                            indices_q[:] = indices_q.size()[0] - 1
                        elif attn_position == 'middle':
                            indices_q[:] = (indices_q.size()[0] + 1) / 2 - 1

                        distance_diff = indices_v - indices_q
                        # print("distance_diff", distance_diff)

                        if attn_type[i] == 'normal':
                            # std = 1 / (attn_param[i] * math.sqrt(2 * math.pi))
                            std = attn_param[i]

                            logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / std) ** 2))
                        else:
                            distance_diff = torch.abs(distance_diff)
                            distance_diff[distance_diff <= attn_param[i]] = 0
                            distance_diff[distance_diff > attn_param[i]] = 1
                            logits = 1 - distance_diff
                            logits = logits / torch.sum(logits, dim=-1, keepdim=True)
                            # logits = F.softmax(logits, dim=-1)
                        self.attn_weights[attn_type[i]][attn_position[i]] = logits
                    else:
                        logits = self.attn_weights[attn_type[i]][attn_position[i]][:queries.shape[1], :values.shape[1]]
                    logits.unsqueeze(0).expand(int(values.shape[0] / self.num_heads),
                                               queries.shape[1],
                                               values.shape[1])
                logits_list.append(logits)
            for l in logits_list:
                print(l.is_cuda)
            logits = torch.stack(logits_list, dim=1)
            # print("logits size", logits.size())
            # print("logits[0]", logits[0])
            # print("logits[1]", logits[1])
            attn_weights = logits.type_as(values)
            # print("attn_weights", attn_weights.size())
            # print("values", values.size())
            attended = torch.bmm(attn_weights # .expand(int(values.shape[0] / self.num_heads),
                                              #         self.num_heads,
                                              #         attn_weights.shape[2],
                                              #         attn_weights.shape[3]).contiguous()
                                 .view(values.shape[0],
                                       attn_weights.shape[2],
                                       attn_weights.shape[3]),
                                 values)

        # print("logits", logits)

        # attn_weights = F.softmax(logits.type_as(values), dim=-1)
        # attn_weights = logits.type_as(values)

        # print("attn_weights", attn_weights)
        # print("attn_weights shape", attn_weights.shape)

        # print("new attention time", time.time() - start)

        # attended = torch.bmm(attn_weights.expand(values.shape[0], attn_weights.shape[0], attn_weights.shape[1]), values)
        print("attended", attended.shape)
        print("attended view", attended.view(
            batch_size,
            self.num_heads,
            -1,
            self.projection_dim
        ).transpose(2, 1).contiguous().view(
            batch_size,
            -1,
            self.num_heads * self.projection_dim
        ).shape)

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
                key_mask=None, attention_mask=None, num_queries=0, layer_i=0):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if 'learned' in self.attn_type:
            if same_tensor(values, keys, queries):
                values, keys, queries = self.project(values, chunks=3)
            elif same_tensor(values, keys):
                values, keys = self.project(values, chunks=2)
                queries, = self.project(queries, 2)
            else:
                values, = self.project(values, 0)
                keys, = self.project(keys, 1)
                queries, = self.project(queries, 2)
        else:
            batch_size = values.shape[0]

            values = F.linear(values, self.input_weights).view(
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


        # print("values", values.shape)
        # print("batch_size", batch_size)
        # print("num_heads", self.num_heads)
        # print("projection_dim", self.projection_dim)

        print("new attention")
        print("values", values.shape)

        attended = self.attention(values, keys, queries, key_mask, attention_mask, layer_i)
        return self.output_projection(attended)
