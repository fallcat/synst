import torch
import random
import numpy as np
import torch.nn.functional as F
from itertools import combinations
from torch.nn import BCELoss
from torch import nn
from torch.distributions import Bernoulli


class LayerMaskPredictor(nn.Module):
    def __init__(self, embedding_size, 
                       num_layers, 
                       lmp_type, 
                       potential_threshold,
                       shuffle_configs,
                       num_configs,
                       loss_func,
                       lmp_eval_mode):
        super(LayerMaskPredictor, self).__init__()

        self.num_layers = num_layers

        self.init_configs(num_layers, num_configs=num_configs, shuffle_configs=shuffle_configs)
        self.potential_threshold = potential_threshold
        self.lmp_type = lmp_type # choose from ['random', 'noskip', 'itertrain']
        self.loss_func = loss_func
        self.eval = lmp_eval_mode

        print("lmp_type", lmp_type)
        print(lmp_type != "random")
        if lmp_type is not "random":
            print("1")
            self.proj1 = nn.Linear(embedding_size, self.all_configs.shape[0]-1)
            if self.loss_func == 'binary_cls':
                self.bce_loss = BCELoss(reduction='none')
            self.reset_parameters()
        else:
            print("2")
            self.sample_distribution = torch.ones((1, 2 * num_layers), device=torch.device("cuda")) * 0.5 # init 0.5

        # print configs for LMP
        print("lmp type : %s" % self.lmp_type)
        print("potential threshold : %f " % self.potential_threshold)
        print("loss func: %s" % self.loss_func)
        print("eval mode : %s" % self.eval)
        print("num of configs: %i" % self.all_configs.shape[0])
        print("shuffle configs: %s" % shuffle_configs)
        print("all-on index: %i" % self.ci_allon)
        
    def init_configs(self, num_layers, num_configs=-1, shuffle_configs=False):

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

        """
        To be deleted later, adding this because oracle result has label shift
        e.g. [0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1.] change to [1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1.]
        1. first layer always 1
        2. decoder layer shift backward by 1
        """
        self.all_configs[:, 0] = 1
        self.all_configs[:, num_layers:] = self.all_configs[:, num_layers-1:-1]
        """
        To be deleted
        """

        self.ci_allon = all_combs.index(tuple(i for i in range(num_layer)))
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
            return torch.ones(lmp_input.size(0), self.num_layers * 2, device=torch.device("cuda"))

        if self.lmp_type == "random":
            dec_sample = np.random.randint(6, 12)
            sample = Bernoulli(self.sample_distribution.expand(lmp_input.size(0), self.num_layers * 2)).sample()
            if sample[dec_sample] != 1:
                sample[dec_sample] = 1
            return sample

        lmp_input = lmp_input.masked_fill_(lmp_input_mask[:, :, None], 0)
        layermask = self.proj1(torch.mean(lmp_input,1))
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
            bs, _, _ = lmp_input.shape
            ret = torch.zeros(layermask.shape[0], self.num_layers * 2, device=torch.device("cuda"))
            max_val, _ = layermask.max(dim=1)
            filtered = (layermask + self.potential_threshold >= max_val[:, None]).float() * self.all_configs_sum_layer[:-1] # all_configs_sum_layer last entry is all-on
            filtered[filtered == 0] = float("inf")
            _, ci = torch.min(filtered, dim=1)
            ci_val = layermask[range(bs), ci]
            # print("{:.2f} {:.2f} {:.2f} {:.2f}".format(ci_val.mean().item(), ci_val.max().item(), ci_val.min().item(), max_val.mean().item() - 2*self.potential_threshold))
            ci[ci_val < max_val.mean().item() - 2*self.potential_threshold] = self.ci_allon
            ret = self.all_configs[ci]

            return ret
