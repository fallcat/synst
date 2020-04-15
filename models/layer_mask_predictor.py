import torch
import random
import pickle
import numpy as np
import pdb
import torch.nn.functional as F
from itertools import combinations
from torch.nn import BCELoss
from torch import nn
from torch.distributions import Bernoulli
from collections import defaultdict


class LayerMaskPredictor(nn.Module):
    def __init__(self, embedding_size, 
                       num_layers, 
                       lmp_type, 
                       potential_threshold,
                       shuffle_configs,
                       num_configs,
                       loss_func,
                       lmp_eval_mode,
                       layermask_file,
                       config_file,
                       random_config):
        super(LayerMaskPredictor, self).__init__()

        self.num_layers = num_layers

        self.init_configs(config_file, num_layers, num_configs=num_configs, shuffle_configs=shuffle_configs)
        self.potential_threshold = potential_threshold
        self.lmp_type = lmp_type # choose from ['random', 'noskip', 'itertrain']
        self.loss_func = loss_func
        self.eval = lmp_eval_mode
        self.layermask_file = layermask_file
        self.random_inference = random_config

        if lmp_type == "random":
            if layermask_file is None:
                self.sample_distribution = torch.ones((1, 2 * num_layers), device=torch.device("cuda")) * 0.5 # init 0.5
            else:
                with open(layermask_file) as layermask_file:
                    self.layermasks = torch.stack([torch.tensor([float(x) for x in list(line.strip())], device=torch.device("cuda")) for line in layermask_file.readlines()])
                    print("Using layermasks from file, layermasks using: ", self.layermasks)
        elif lmp_type == "lengths":
            all_combs = sum([list(combinations(range(num_layers * 2), k)) for k in range(1, num_layers * 2 + 1)], [])
            all_combs = [x for x in all_combs if any(y >= num_layers for y in x)]
            all_combs_array = np.zeros((len(all_combs), num_layers * 2))
            for i, comb in enumerate(all_combs):
                all_combs_array[i, np.array(comb)] = 1
            combs_by_k = defaultdict(list)
            for i, comb in enumerate(all_combs_array):
                k = np.sum(comb)
                combs_by_k[k].append(torch.tensor(comb))
            with open(layermask_file) as layermask_file:
                self.cut_offs = torch.tensor([int(x) for x in layermask_file.readline().strip().split()]).unsqueeze(0)
            bins = self.cut_offs.shape[1] + 1
            bin_width = 2 * num_layers / bins
            self.combs_by_bin = defaultdict(list)
            for k in combs_by_k:
                print("k", k, "np.floor((k - 1) / bin_width)", np.floor((k - 1) / bin_width))
                self.combs_by_bin[np.floor((k - 1) / bin_width)].extend(combs_by_k[k])
            print("self.combs_by_bin", self.combs_by_bin)
        elif "ensemble" in lmp_type:
            if layermask_file is None:
                raise Exception("No layermask found for ensemble")
            else:
                with open(layermask_file) as layermask_file:
                    self.layermasks = torch.stack(
                        [torch.tensor([float(x) for x in list(line.strip())], device=torch.device("cuda")) for line in
                         layermask_file.readlines()])
                    print("Using layermasks from file, layermasks using: ", self.layermasks)
        else:
            self.proj1 = nn.Linear(embedding_size, self.all_configs.shape[0] - 1)
            if self.loss_func == 'binary_cls':
                self.bce_loss = BCELoss(reduction='none')
            self.reset_parameters()


        # print configs for LMP
        print("lmp type : %s" % self.lmp_type)
        print("potential threshold : %f " % self.potential_threshold)
        print("loss func: %s" % self.loss_func)
        print("eval mode : %s" % self.eval)
        print("num of configs: %i" % self.all_configs.shape[0])
        print("shuffle configs: %s" % shuffle_configs)
        print("all-on index: %i" % self.ci_allon)
        print("random inference: %s" % self.random_inference)

    def get_lmp_type(self):
        return self.lmp_type

    def get_layermasks(self):
        if self.layermasks is None:
            raise Exception("No layermasks defines")
        return self.layermasks

    def init_configs(self, config_file, num_layers, num_configs=-1, shuffle_configs=False):

        if config_file is None:
            num_layer = 2 * num_layers
            all_combs = sum([list(combinations(range(num_layer), k)) for k in range(1, num_layer+1)], [])
            all_combs = [x for x in all_combs if any(y >= num_layer//2 for y in x)]

            if shuffle_configs:
                random.Random(42).shuffle(all_combs)

            if num_configs != -1:
                all_combs = all_combs[:num_configs]
                all_combs.append(tuple(i for i in range(num_layer)))

            self.all_configs = torch.zeros(len(all_combs), num_layer, device=torch.device("cuda"))
            for ci, c in enumerate(all_combs):
                for cii in c:
                    self.all_configs[ci, cii] += 1
            self.ci_allon = all_combs.index(tuple(i for i in range(num_layer)))
            self.all_configs_sum_layer = self.all_configs.sum(dim=1) # len(all_combs) x 1
        else:
            with open(config_file, 'rb') as f:
                configs = pickle.load(f)['configs']

            self.all_configs = torch.tensor(configs, device=torch.device("cuda")).float()
            self.ci_allon = self.all_configs.shape[0] - 1
            self.all_configs_sum_layer = self.all_configs.sum(dim=1) # len(all_combs) x 1

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.proj1.weight, gain)
        nn.init.constant_(self.proj1.bias, 0.)

    def forward(self, lmp_input, lmp_input_mask, aggregate_stats=None):
        '''
            lmp_input: [bs, L, embedding_size]
            layermask: [bs, 2*num_layers]
            return: sampled layermask, raw-layermask-distribution
        '''
        
        # special case: not skipping
        if self.lmp_type == "noskip":
            return torch.ones(self.num_layers * 2, device=torch.device("cuda"))

        batch_size = lmp_input.size(0)

        if self.lmp_type == "random":
            if self.layermask_file is None:
                sample = Bernoulli(self.sample_distribution.expand(batch_size, self.num_layers * 2)).sample()
                violate_indices = torch.sum(sample[:, self.num_layers: self.num_layers * 2], dim=1) == 0
                dec_sample_size = violate_indices.sum()
                if dec_sample_size > 0:
                    dec_sample = np.random.randint(self.num_layers, self.num_layers * 2, size=dec_sample_size)
                    sample[violate_indices, dec_sample] = 1
                return sample
            else:
                indices = torch.multinomial(torch.ones(self.layermasks.size(0)), batch_size, replacement=True)
                return self.layermasks[indices]

        print("lmp_input_mask", lmp_input_mask)
        if self.lmp_type == "lengths":
            cut_offs = ((self.cut_offs - lmp_input_mask[lmp_input_mask == 0].sum(1).unsqueeze(1)) >= 0)
            bins = [(cut_off == 1).nonzero()[0] for cut_off in cut_offs]
            print("bins", bins)
            return torch.stack([random.choice(self.combs_by_bin[b]) for b in bins])

        if self.lmp_type == "ensemble":
            return self.layermasks  # .unsqueeze(0).expand(batch_size, -1)

        if self.lmp_type == "ensemble_total":
            layermasks_shape = self.layermasks.shape
            return self.layermasks.unsqueeze(0)\
                .expand(int(batch_size / layermasks_shape[0]),
                        layermasks_shape[0],
                        layermasks_shape[1]).contiguous().view(batch_size,
                                                              layermasks_shape[1])

        lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
        layermask = self.proj1(torch.sum(lmp_input,1))
        layermask = torch.sigmoid(layermask)

        if not self.eval:
            if self.loss_func == 'binary_cls':
                loss = self.bce_loss(layermask, aggregate_stats)
                loss = loss.mean(dim=1).mean()
                return loss

            elif self.loss_func == "regr":
                loss = ((layermask - aggregate_stats)**2).mean(dim=1).mean()
                return loss

            elif self.loss_func == "scaled_regr":
                loss = ((layermask - aggregate_stats)**2).mean(dim=1).mean()
                return loss

            elif self.loss_func == "rank":
                raise NotImplementedError

            else:
                raise NotImplementedError
        else:

            if not self.random_inference:
                bs, _, _ = lmp_input.shape
                max_val, _ = layermask.max(dim=1)
                filtered = (layermask + self.potential_threshold >= max_val[:, None]).float() * self.all_configs_sum_layer[:-1] # all_configs_sum_layer last entry is all-on
                filtered[filtered == 0] = float("inf")
                _, ci = torch.min(filtered, dim=1)
                ci_val = layermask[range(bs), ci]
                ci[ci_val < max_val.mean().item() - 2*self.potential_threshold] = self.ci_allon
                ret = self.all_configs[ci]

            else:
                ci = [random.randint(0,self.ci_allon) for i in range(bs)]
                ret = self.all_configs[ci]

            return ret
