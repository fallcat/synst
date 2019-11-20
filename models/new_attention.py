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
        self.attn_threshold = attn_config['attn_threshold']
        self.attn_window = attn_config['attn_window']
        self.half_window = int((self.attn_window - 1) / 2)
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
        # self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))

        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()
        self.attn_configs = list(self.load_attn_configs())

        self.attn_weights = {}
        self.times = {}

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

            attn_configs_names = ['attn_type', 'attn_position', 'attn_param', 'attn_displacement', 'attn_threshold']

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

            easy_positions = ['left', 'center', 'right', 'first']

            with torch.no_grad():
                if ('normal' == attn_configs[0] or 'normal' == set(attn_configs[0])) \
                        and (attn_configs[1] in easy_positions or set(attn_configs[1]).issubset(easy_positions)):
                    attn_type, attn_position, attn_param, attn_displacement = attn_configs

                    if list not in [type(x) for x in [attn_displacement, attn_param]]:
                        distance_diff = torch.arange(-self.half_window, self.half_window + 1, dtype=torch.float32)
                        conv_filter = (1 / (attn_param * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / attn_param) ** 2)).view(1, 1, -1)
                    else:
                        attn_config = []
                        for attn_config_i in [attn_type, attn_position, attn_param, attn_displacement]:
                            if type(attn_config_i) is not list:
                                attn_config.append([attn_config_i] * self.num_heads)
                            else:
                                attn_config.append(attn_config_i)

                        attn_type, attn_position, attn_param, attn_displacement = attn_config
                        conv_filter = []
                        for i in range(self.num_heads):
                            distance_diff = torch.arange(-self.half_window, self.half_window + 1, dtype=torch.float32)
                            conv_filter.append((1 / (attn_param[i] * math.sqrt(2 * math.pi)) * torch.exp(
                                - 1 / 2 * (distance_diff / attn_param[i]) ** 2))).view(1, 1, -1)
                    if type(attn_position) is not list:
                        attn_position = [attn_position]
                    mask_conv_filters = []
                    print("attn_configs", attn_configs)
                    print("attn_position", attn_position)
                    for i, p in enumerate(attn_position):
                        mask_conv_filter = conv_filter.clone()
                        d = attn_displacement[i] if type(attn_displacement) is list else attn_displacement
                        if p in ["center", "first"]:
                            mask_conv_filter[:, :, -self.half_window:] = 0
                        elif p == "left":
                            mask_conv_filter[:, :, self.attn_window - self.half_window + d:] = 0
                        else:
                            mask_conv_filter[:, :, -self.half_window - d:] = 0
                        mask_conv_filters.append(mask_conv_filter)
                        print("mask_conv_filters first calculated", mask_conv_filters)
                else:
                    conv_filter = None

            yield attn_configs, conv_filter, mask_conv_filters

    def attention(self, values, keys, queries, key_mask=None, mask=None, layer_i=0, decoder_position=-1, input_lens=None, learned=False):
        ''' Scaled dot product attention with optional masks '''
        queries_shape = queries.shape
        values_shape = values.shape

        # By this point the values, keys, and queries all have B * H as their first dimension
        batch_size = queries_shape[0] // self.num_heads

        # Get the parameters for hard-coded attention for this layer, which was already initialized
        attn_configs, conv_filter, mask_conv_filters = self.attn_configs[layer_i]
        print("conv_filter", conv_filter)
        attn_type, attn_position, attn_param, attn_displacement = attn_configs
        print("attn_type, attn_position, attn_param, attn_displacement", attn_type, attn_position, attn_param, attn_displacement)

        # If we are using learned attention, then just do it the same way as multi-headed attention
        if attn_type == 'learned' or learned:
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

        # If we are learning some of the heads but not all, apply multiheaded attention to thoese heads together and
        # concat with rest of the hard-coded heads later.
        elif 'learned' in attn_type:
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

        # If we have conv filter, then we don't need to go through the huge amount of calculation
        # but can just use conv filter
        # conv_filter = None
        old_values = values
        if conv_filter is not None:
            # print("hi")
            if list not in [type(x) for x in [attn_param, attn_displacement]]:
                # print("hello")
                # if attn_position in ['center', 'first']:
                #     padding = self.half_window
                # else:
                #     padding = self.half_window + attn_displacement
                # print("Using CNN!")
                if mask is not None:
                    # print("values", values.shape)
                    # print("mask", mask.shape)
                    # print("conv_filter", conv_filter.shape)
                    use_conv_filter = mask_conv_filters if len(mask_conv_filters) != 1 else mask_conv_filters[0]
                    print("using mask")
                    # print("conv_filter", conv_filter)
                    # values = values * (mask == 0).to(dtype=torch.float32)
                else:
                    use_conv_filter = conv_filter
                print("use_conv_filter", use_conv_filter)
                if key_mask is not None:
                    values = values.view(batch_size, self.num_heads, values_shape[1], values_shape[2])
                    # print("key_mask", key_mask.shape)
                    # print("key_mask[:, None]", key_mask[:, None, :, None].shape)
                    # print("values", values.shape)
                    values.masked_fill_(key_mask[:, None, :, None], float(0))
                    values = values.view(values_shape)

                values = values.transpose(1, 2).contiguous().view(batch_size * self.embed_dim, 1, -1)
                try:
                    if type(use_conv_filter) is not list:
                        attended = F.conv1d(values, use_conv_filter, padding=self.half_window + attn_displacement)
                    else:
                        attended = []
                        for i, f in enumerate(use_conv_filter):
                            a = F.conv1d(values.view(batch_size,
                                                     self.num_heads,
                                                     self.projection_dim,
                                                     -1)[:, i].contiguous().view(batch_size * self.projection_dim,
                                                                                 1,
                                                                                 -1),
                                         use_conv_filter[i], padding=self.half_window + attn_displacement)
                            attended.append(a.view(batch_size, self.projection_dim, -1))
                        attended = torch.stack(attended, dim=1)
                except:
                    # print("Convert conv filter to correct device")
                    if values.is_cuda:
                        if type(use_conv_filter) is not list:
                            use_conv_filter = use_conv_filter.cuda().type_as(values).to(values.get_device())
                        else:
                            use_conv_filter = [x.cuda().type_as(values).to(values.get_device()) for x in use_conv_filter]
                    # use_conv_filter.type_as(values).to(values.get_device())
                    if mask is None:
                        self.attn_configs[layer_i] = attn_configs, use_conv_filter, mask_conv_filters
                    else:
                        if type(use_conv_filter) is not list:
                            self.attn_configs[layer_i] = attn_configs, conv_filter, [use_conv_filter]
                        else:
                            self.attn_configs[layer_i] = attn_configs, conv_filter, [use_conv_filter]
                    # print("values.get_device()", values.get_device())
                    # print("conv_filter type", type(conv_filter))
                    # print("conv_filter", conv_filter.is_cuda)
                    # attended = F.conv1d(values, use_conv_filter, padding=self.half_window + attn_displacement)
                    if type(use_conv_filter) is not list:
                        attended = F.conv1d(values, use_conv_filter, padding=self.half_window + attn_displacement)
                    else:
                        attended = []
                        for i, f in enumerate(use_conv_filter):
                            a = F.conv1d(values.view(batch_size,
                                                     self.num_heads,
                                                     self.projection_dim,
                                                     -1)[:, i].contiguous().view(batch_size * self.projection_dim,
                                                                                 1,
                                                                                 -1),
                                         use_conv_filter[i], padding=self.half_window + attn_displacement)
                            attended.append(a)
                        attended = torch.stack(attended, dim=1)
                attended = attended.view(batch_size, self.num_heads,
                                         self.projection_dim,
                                         -1).transpose(2, 3).contiguous()
                if self.word_count_ratio == 1:
                    if attended.shape[3] < queries_shape[1] + 2 * attn_displacement:
                        new_attended = values.new_zeros((queries_shape[0],
                                                        queries_shape[1] + 2 * attn_displacement,
                                                        queries_shape[2])).view(batch_size,
                                                                               self.num_heads,
                                                                               -1,
                                                                               self.projection_dim)
                        new_attended[:, :, :attended.shape[3]] = attended
                        attended = new_attended
                    # if values_shape[1] >= queries_shape[1]:
                        # print("greater")
                        # print("values_shape[1]", values_shape[1])
                        # print("queries_shape[1]", queries_shape[1])

                    if type(attn_position) is not list:
                        if attn_position == "center":
                            conv_attended = attended[:, :, attn_displacement:queries_shape[1] + attn_displacement]
                        elif attn_position == "left":
                            conv_attended = attended[:, :, :queries_shape[1]]
                        elif attn_position == "right":
                            conv_attended = attended[:, :, 2*attn_displacement:queries_shape[1] + 2*attn_displacement]
                        else:
                            conv_attended = attended[:, :, attn_displacement:attn_displacement+1].expand(batch_size, self.num_heads, queries_shape[1], self.projection_dim)
                    else:
                        conv_attended = []
                        for i, p in enumerate(attn_position):
                            if attn_position == "center":
                                conv_attended.append(attended[:, i,
                                                attn_displacement:queries_shape[1] + attn_displacement])
                            elif attn_position == "left":
                                conv_attended.append(attended[:, i, attn_displacement:queries_shape[1]])
                            elif attn_position == "right":
                                conv_attended.append(attended[:, i,
                                                     attn_displacement:queries_shape[1] + 2 * attn_displacement])
                            else:
                                conv_attended.append(attended[:, i:i+1, attn_displacement:attn_displacement+1].expand(batch_size,
                                                                                              queries_shape[1],
                                                                                              self.projection_dim))
                        conv_attended = torch.stack(conv_attended, dim=1)
                    conv_attended = conv_attended.view(batch_size,
                                                       self.num_heads,
                                                       -1,
                                                       self.projection_dim
                                                       ).transpose(2, 1).contiguous().view(batch_size,
                                                                                           -1,
                                                                                           self.num_heads * self.projection_dim
                                                                                           )
                    # else:
                    #     new_attended = values.new_zeros(queries_shape)
                    #     new_attended[:, :values_shape[1]] = attended
                    #     conv_attended = new_attended
                else:
                    if attended.shape[3] < round(queries_shape[1] * self.word_count_ratio) + 2 * attn_displacement:
                        new_attended = values.new_zeros((queries_shape[0],
                                                         queries_shape[1] * self.word_count_ratio
                                                         + 2 * attn_displacement,
                                                         queries_shape[2])).view(batch_size,
                                                                                 self.num_heads,
                                                                                 -1,
                                                                                 self.projection_dim)
                        new_attended[:, :, :attended.shape[3]] = attended
                        attended = new_attended

                        # print(
                        #         "torch.round(torch.arange(queries_shape[1], device=values.get_device()) * self.word_count_ratio)",
                        #         torch.round(torch.arange(attn_displacement, queries_shape[1] + attn_displacement,
                        #                                  device=values.get_device(),
                        #                                  dtype=torch.float32) * self.word_count_ratio))

                    indices_q = torch.round(torch.arange(queries_shape[1],
                                                         device=values.get_device(),
                                                         dtype=torch.float32) * self.word_count_ratio).long()
                    if type(attn_position) is not list:
                        if attn_position == "center":
                            conv_attended = attended[:, :, indices_q + attn_displacement]
                        elif attn_position == "left":
                            conv_attended = attended[:, :, indices_q]
                        elif attn_position == "right":
                            conv_attended = attended[:, :, indices_q + 2 * attn_displacement]
                        else:
                            conv_attended = attended[:, :, attn_displacement:attn_displacement+1].expand(batch_size, self.num_heads, queries_shape[1], self.projection_dim)
                    else:
                        conv_attended = []
                        for i, p in enumerate(attn_position):
                            if attn_position == "center":
                                conv_attended.append(attended[:, i, indices_q])
                            elif attn_position == "left":
                                conv_attended.append(attended[:, i, indices_q - attn_displacement])
                            elif attn_position == "right":
                                conv_attended.append(attended[:, i, indices_q + attn_displacement])
                            else:
                                conv_attended.append(attended[:, i:i+1, attn_displacement:attn_displacement+1].expand(batch_size,
                                                                                              queries_shape[1],
                                                                                              self.projection_dim))
                        conv_attended = torch.stack(conv_attended, dim=1)

                    conv_attended = conv_attended.view(batch_size,
                                                       self.num_heads,
                                                       -1,
                                                       self.projection_dim
                                                       ).transpose(2, 1).contiguous().view(batch_size,
                                                                                           -1,
                                                                                           self.num_heads * self.projection_dim
                                                                                           )
                        # torch.index_select(attended, 1, indices_q)
                    # else:
                    #     new_attended = values.new_zeros(queries_shape)
                    #     new_attended[:, :values_shape[1]] = attended
                    #     conv_attended = new_attended

        # If we want to look at last token of the sentence, or different bins of the sentence,
        # we would need sentence length to compute the focused position. If we have input_lens,
        # it means we are in training, and we can directly use input_lens - 1. If we don't have input lens,
        # we can use key_mask to compute it, but it's a bit slower. At test time, we simply use the length
        # of the whole sentence.

        values = old_values

        with torch.no_grad():

            if not {'last', 'bin'}.isdisjoint(attn_position) or attn_position in ['last', 'bin']:
                if input_lens is not None:
                    last_indices = (input_lens - 1).cpu().view(-1)
                elif key_mask is not None:
                    last_indices = ((key_mask == 0).sum(dim=1) - 1).view(-1)
                else:
                    last_indices = torch.tensor([values_shape[1] - 1] * queries_shape[0]).view(-1).type_as(values)

            # If every arg is not list, then we can compute all the heads together as they are all the same
            if list not in [type(x) for x in [attn_type, attn_position, attn_param, attn_displacement]]:
                need_recompute = False
                # We check if we have already stored attention of this size or larger in the dictionary self.attn_weights.
                # If we have, then we don't need to recompute, but can just retrieve from the dictionary.
                # If we haven't, then we need to recompute and store in the dict, and then retrieve from the dictionary.
                if attn_type not in self.attn_weights:
                    self.attn_weights[attn_type] = {}
                if attn_position not in self.attn_weights[attn_type]:
                    self.attn_weights[attn_type][attn_position] = {}
                if attn_position == 'center':
                    if attn_param not in self.attn_weights[attn_type][attn_position] \
                            or (queries_shape[1] > self.attn_weights[attn_type][attn_position][attn_param].shape[0]
                                or decoder_position + 1 > self.attn_weights[attn_type][attn_position][attn_param].shape[0]
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
                                    or decoder_position + 1 > self.attn_weights[attn_type][attn_position][attn_param][attn_displacement].shape[0]
                                    or values_shape[1] > self.attn_weights[attn_type][attn_position][attn_param][attn_displacement].shape[1]):
                            need_recompute = True
                    else:  # attn_position in ['last', 'bin']
                        max_last_index = last_indices[0]
                        if attn_position == 'last':
                            if attn_param not in self.attn_weights[attn_type][attn_position] \
                                    or max_last_index + 1 > \
                                    self.attn_weights[attn_type][attn_position][attn_param].shape[0]:
                                need_recompute = True
                        else:
                            if attn_param not in self.attn_weights[attn_type][attn_position]:
                                self.attn_weights[attn_type][attn_position][attn_param] = {}
                                need_recompute = True
                            elif attn_displacement not in self.attn_weights[attn_type][attn_position][
                                attn_param] or \
                                    max_last_index + 1 > self.attn_weights[attn_type][attn_position][attn_param][
                                attn_displacement].shape[0]:
                                need_recompute = True

                if need_recompute:
                    indices_v = torch.arange(values_shape[1]).view(1, -1).type_as(values)

                    # If attention is not looking at last or bin, we don't need to know sentence length
                    if attn_position not in ['last', 'bin']:
                        # If looking at the first token, we just need one vector, and use the first l tokens for each length
                        if attn_position == 'first':
                            indices_q = torch.tensor(0.0).type_as(values) # torch.full((queries_shape[1], 1), 0).to(dtype=torch.float32)
                        # If it is training time, or encoder self attention at test time,
                        # we compute the whole matrix with attention focused on the diagonal
                        elif decoder_position == -1:
                            indices_q = torch.round(torch.arange(queries_shape[1]
                                                                 ).view(-1, 1).type_as(values) * self.word_count_ratio)
                            print("indices_q", indices_q)
                        # If it is test time decoder self/source attention, we compute the matrix of size of this sentence
                        else:
                            indices_q = torch.round(torch.arange(decoder_position + 1
                                                                 ).view(-1, 1).type_as(values) * self.word_count_ratio)
                        # If we are looking at left or right, we can move the center according to the offset we specify
                        if attn_position == 'left':
                            indices_q = indices_q - attn_displacement
                        elif attn_position == 'right':
                            indices_q = indices_q + attn_displacement

                        # This is the distance from center of distribution for each position
                        distance_diff = indices_v - indices_q

                    # If the attention is looking at the last indices or different bins, we compute a matrix with attention
                    # of all sentence lengths until that length
                    else:
                        indices_q = torch.arange(max_last_index + 1).view(-1, 1).type_as(values)
                        old_indices_q = indices_q
                        # If we are looking at bin, then we calculate the center of nth bin: n/l of the sentence
                        if attn_position == 'bin':
                            ratio = (attn_displacement - 0.5) / self.attn_bins
                            indices_q = -0.5 + indices_q * ratio
                        # This is the distance from center of distribution for each position
                        distance_diff = (indices_v - indices_q)

                    # Compute the distribution with the normal distribution's formula
                    if attn_type == 'normal':
                        std = attn_param
                        logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / std) ** 2))
                        if self.attn_threshold > 0:
                            logits[logits < self.attn_threshold] = 0
                    # Compute uniform distribution within a window
                    else:
                        if attn_param < 0 and attn_position == 'bin':
                            attn_param_curr = (0.5 * old_indices_q / self.attn_bins).view(-1, 1)
                        else:
                            attn_param_curr = attn_param
                        distance_diff = torch.abs(distance_diff)
                        distance_diff[distance_diff <= attn_param_curr] = 0
                        distance_diff[distance_diff > attn_param_curr] = 1
                        logits = 1 - distance_diff
                        logits_sum = torch.sum(logits, dim=-1, keepdim=True)
                        logits_sum[logits_sum == 0] = 1
                        logits = logits / logits_sum

                    # Store the attention weights
                    if attn_position in ['center', 'first', 'last']:
                        self.attn_weights[attn_type][attn_position][attn_param] = logits
                    else:
                        self.attn_weights[attn_type][attn_position][attn_param][attn_displacement] = logits

                # Retrieve attention weights
                if attn_position in ['center', 'first', 'last']:
                    retrieve_dict = self.attn_weights[attn_type][attn_position][attn_param]
                else:
                    retrieve_dict = self.attn_weights[attn_type][attn_position][attn_param][attn_displacement]

                if attn_position in ['center', 'first', 'left', 'right']:
                    if decoder_position == -1:
                        logits = retrieve_dict[:queries_shape[1], :values_shape[1]].unsqueeze(0).unsqueeze(0)
                    else:
                        if attn_position == 'first':
                            logits = retrieve_dict[:, :values_shape[1]].view(1, 1, 1, -1)
                        else:
                            logits = retrieve_dict[decoder_position, :values_shape[1]].view(1, 1, 1, -1)
                else:
                    if decoder_position == -1:
                        logits = torch.index_select(retrieve_dict, 0, last_indices)[:, :values_shape[1]].unsqueeze(
                            1).unsqueeze(1)
                    else:
                        logits = torch.index_select(retrieve_dict, 0, last_indices)[max_last_index, :values_shape[1]].view(
                            1, 1, 1, -1)

                # Copy the weights to each head
                attn_weights = logits.expand(batch_size, self.num_heads, queries_shape[1], values_shape[1])\
                    .contiguous().view(-1,
                                       queries_shape[1],
                                       values_shape[1])

            # If one of the attention parameters is list (different in different heads), then make all of them lists,
            # and compute one head by one head, and stack them together.
            else:
                attn_config = []
                for attn_config_i in [attn_type, attn_position, attn_param, attn_displacement]:
                    if type(attn_config_i) is not list:
                        attn_config.append([attn_config_i] * self.num_heads)
                    else:
                        attn_config.append(attn_config_i)

                attn_type, attn_position, attn_param, attn_displacement = attn_config

                logits_list = []

                for i in range(self.num_heads):
                    if attn_type[i] == 'learned':
                        logits = logits_[:, learned_count]
                        learned_count += 1
                    else:
                        need_recompute = False
                        # We check if we have already stored attention of this size or larger in the dictionary self.attn_weights.
                        # If we have, then we don't need to recompute, but can just retrieve from the dictionary.
                        # If we haven't, then we need to recompute and store in the dict, and then retrieve from the dictionary.
                        if attn_type[i] not in self.attn_weights:
                            self.attn_weights[attn_type[i]] = {}
                        if attn_position[i] not in self.attn_weights[attn_type[i]]:
                            self.attn_weights[attn_type[i]][attn_position[i]] = {}
                        if attn_position[i] == 'center':
                            if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]] \
                                    or (queries_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[0]
                                        or decoder_position + 1 > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[0]
                                        or values_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[
                                            1]):
                                need_recompute = True
                        elif attn_position[i] == 'first':
                            if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]] \
                                    or values_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[1]:
                                need_recompute = True
                        else:
                            # If attention is not looking at last or bin, we don't need to know sentence length
                            if attn_position[i] in ['left', 'right']:
                                if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]]:
                                    self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = {}
                                    need_recompute = True
                                    # if self.which_attn == 'decoder':
                                    #     print("left, not exist")
                                if attn_displacement[i] not in self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] \
                                        or (queries_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]].shape[0]
                                            or decoder_position + 1 > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]].shape[0]
                                            or values_shape[1] > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][
                                                attn_displacement[i]].shape[1]):
                                    need_recompute = True
                            else:  # attn_position[i] in ['last', 'bin']
                                max_last_index = last_indices[0]
                                if attn_position[i] == 'last':
                                    if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]] \
                                            or max_last_index + 1 > \
                                            self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]].shape[0]:
                                        need_recompute = True
                                else:
                                    if attn_param[i] not in self.attn_weights[attn_type[i]][attn_position[i]]:
                                        self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = {}
                                        need_recompute = True
                                    elif attn_displacement[i] not in self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] or \
                                            max_last_index + 1 > self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]].shape[0]:
                                        need_recompute = True

                        if need_recompute:
                            indices_v = torch.arange(values_shape[1]).view(1, -1).type_as(values)

                            # If attention is not looking at last or bin, we don't need to know sentence length
                            if attn_position[i] not in ['last', 'bin']:
                                # If looking at the first token, we just need one vector,
                                # and use the first l tokens for each length
                                if attn_position[i] == 'first':
                                    indices_q = torch.tensor(0.0).type_as(values)
                                # If it is training time, or encoder self attention at test time,
                                # we compute the whole matrix with attention focused on the diagonal
                                elif decoder_position == -1:
                                    indices_q = torch.round(torch.arange(queries_shape[1]
                                                             ).view(-1, 1).type_as(values) * self.word_count_ratio)
                                # If it is test time decoder self/source attention,
                                # we compute the matrix of size of this sentence
                                else:
                                    indices_q = torch.round(torch.arange(decoder_position + 1
                                                             ).view(-1, 1).type_as(values) * self.word_count_ratio)
                                # If we are looking at left or right,
                                # we can move the center according to the offset we specify
                                if attn_position[i] == 'left':
                                    indices_q = indices_q - attn_displacement[i]
                                elif attn_position[i] == 'right':
                                    indices_q = indices_q + attn_displacement[i]

                                # This is the distance from center of distribution for each position
                                distance_diff = indices_v - indices_q

                            # If the attention is looking at the last indices, need to take masks into consideration
                            else:
                                indices_q = torch.arange(max_last_index + 1).view(-1, 1).type_as(values)
                                old_indices_q = indices_q
                                # If we are looking at bin, then we calculate the center of nth bin: n/l of the sentence
                                if attn_position[i] == 'bin':
                                    ratio = (attn_displacement[i] - 0.5) / self.attn_bins
                                    indices_q = -0.5 + indices_q * ratio

                                # This is the distance from center of distribution for each position
                                distance_diff = (indices_v - indices_q)

                            # Compute the distribution with the normal distribution's formula
                            if attn_type[i] == 'normal':
                                std = attn_param[i]
                                logits = (1 / (std * math.sqrt(2 * math.pi)) * torch.exp(
                                    - 1 / 2 * (distance_diff / std) ** 2))
                                if self.attn_threshold > 0:
                                    logits[logits < self.attn_threshold] = 0
                            # Compute uniform distribution within a window
                            else:
                                if attn_param[i] < 0 and attn_position[i] == 'bin':
                                    attn_param_curr = (0.5 * old_indices_q / self.attn_bins).view(-1, 1)
                                else:
                                    attn_param_curr = attn_param[i]
                                distance_diff = torch.abs(distance_diff)
                                distance_diff[distance_diff <= attn_param_curr] = 0
                                distance_diff[distance_diff > attn_param_curr] = 1
                                logits = 1 - distance_diff
                                logits_sum = torch.sum(logits, dim=-1, keepdim=True)
                                logits_sum[logits_sum == 0] = 1
                                logits = logits / logits_sum

                            # Store the attention weights
                            if attn_position[i] in ['center', 'first', 'last']:
                                self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]] = logits
                            else:
                                self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][attn_displacement[i]] = logits

                        # Retrieve attention weights
                        if attn_position[i] in ['center', 'first', 'last']:
                            retrieve_dict = self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]]
                        else:
                            retrieve_dict = self.attn_weights[attn_type[i]][attn_position[i]][attn_param[i]][
                                attn_displacement[i]]

                        if attn_position[i] in ['center', 'first', 'left', 'right']:
                            if decoder_position == -1:
                                logits = retrieve_dict[:queries_shape[1], :values_shape[1]].unsqueeze(0).unsqueeze(0)
                            else:
                                if attn_position[i] == 'first':
                                    logits = retrieve_dict[:, :values_shape[1]].view(1, 1, 1, -1)
                                else:
                                    # print("attn_position[i]", attn_position[i])
                                    # print("retrieve_dict", retrieve_dict)
                                    logits = retrieve_dict[decoder_position, :values_shape[1]].view(1, 1, 1, -1)
                        else:
                            if decoder_position == -1:
                                logits = torch.index_select(retrieve_dict, 0, last_indices)[:, :values_shape[1]].unsqueeze(1).unsqueeze(1)
                            else:
                                logits = retrieve_dict[max_last_index, :values_shape[1]].view(1, 1, 1, -1)

                        # Expand the logits to the same size to stack with other heads together later
                        logits = logits.expand(batch_size, 1, queries_shape[1], values_shape[1])  # .type_as(values)

                    logits_list.append(logits)
                attn_weights = torch.stack(logits_list, dim=1)
                attn_weights = attn_weights.view(values_shape[0],
                                                 queries_shape[1],
                                                 values_shape[1])

        if mask is not None:
            try:
                attn_weights = attn_weights * (mask == 0).to(dtype=torch.float32)
            except:
                attn_weights = attn_weights.to(mask.device)
                attn_weights = attn_weights * (mask == 0).to(dtype=torch.float32)
        if key_mask is not None:
            attn_weights_shape = attn_weights.shape
            # print("previous implementation")
            # print("attn_weights_shape", attn_weights_shape)
            # print("key_mask", key_mask.shape)
            # print("key_mask[:, None, None]", key_mask[:, None, None].shape)
            batch_size = attn_weights_shape[0] // self.num_heads
            attn_weights = attn_weights.view(batch_size, self.num_heads, attn_weights_shape[1], attn_weights_shape[2])
            try:
                attn_weights.masked_fill_(key_mask[:, None, None], float(0))
            except:
                attn_weights = attn_weights.to(key_mask.device)
                
                attn_weights.masked_fill_(key_mask[:, None, None], float(0))
            attn_weights = attn_weights.view(attn_weights_shape)

        attended = torch.bmm(attn_weights,
                             values)

        # print("value", values_shape[1])
        # print("query", queries_shape[1])
        #
        # print("conv_attended", conv_attended.shape)
        # print("attended", attended.shape)

        same = (attended.view(
            batch_size,
            self.num_heads,
            -1,
            self.projection_dim
        ).transpose(2, 1).contiguous().view(
            batch_size,
            -1,
            self.num_heads * self.projection_dim
        ) == conv_attended)

        print("self.which_attn", self.which_attn)
        print("same", torch.sum(same == 0))
        # if torch.sum(same == 0).item() != 0:
        #     torch.set_printoptions(profile='full')
            # print("conv_attended", conv_attended)
            # print("attended", attended)

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
