'''
SynST
--
Main entry point for translating from SynST
'''

from __future__ import print_function

import os
import sys
import timeit
from contextlib import ExitStack

import torch
from tqdm import tqdm

from utils import profile
from utils import tqdm_wrap_stdout

from itertools import combinations

import sacrebleu
from collections import defaultdict
import json
import datetime
import pdb

class Translator(object):
    ''' An object that encapsulates model evaluation '''
    CURRENT = None

    def __init__(self, config, model, dataloader, device):
        self.config = config
        self.dataloader = dataloader
        self.translator = model.translator(config).to(device)

        self.modules = {
            'model': model
        }

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    @property
    def annotation_sos_idx(self):
        ''' Get the annotation sos index '''
        return self.dataset.sos_idx - self.dataset.reserved_range

    @property
    def annotation_eos_idx(self):
        ''' Get the annotation eos index '''
        return self.dataset.eos_idx - self.dataset.reserved_range

    @property
    def padding_idx(self):
        ''' Get the padding index '''
        return self.dataset.padding_idx

    def instantiate_combination(self, total_num_layer):

        all_comb = {k : torch.tensor(list(combinations(range(total_num_layer), k))).cuda() for k in range(1, total_num_layer+1)}
        ret = {k: torch.zeros(len(all_comb[k]), total_num_layer) for k in all_comb}

        def assign(ret_k, comb_k, k):
            len_comb_k = len(comb_k)
            indices_x = torch.arange(len_comb_k).unsqueeze(1).expand(len_comb_k, k) # generate all indices_X
            ret_k[indices_x, comb_k] = 1
            ret_k = ret_k[ret_k[:, -total_num_layer//2:].sum(dim=1) >= 1] # select #dec >= 1
            return ret_k

        ret = {k: assign(ret[k], all_comb[k], k) for k in ret}

        return ret

    def translate_all(self, output_file, epoch, experiment, verbose=0):
        ''' Generate all predictions from the dataset '''
        def get_description():
            description = f'Generate #{epoch}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        batches = self.dataloader

        # generate all combinations
        total_num_layer = len(self.modules['model'].encoders) * 2
        #self.modules['model'].set_LMP_type('random')
        all_combinations = {8: torch.tensor([[0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1.]])}
        # all_combinations = self.instantiate_combination(total_num_layer) # {k: [[0,1,0,0...], [1,0,0,0...]]}

        ordered_outputs = []
        for i, batch in enumerate(batches):
            if i == 0:
                break
        # batch = batches[self.config.example_id]

        # num bpe tokens
        num_src_bpe_tokens = batch['input_lens']
        target_text = ' '.join(self.dataset.decode(batch['targets'][0].numpy().tolist(), trim=not verbose))
        combination = torch.tensor([[1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1.]])
        sequences, _ = self.translator.translate(batch, raw_layermask=combination)
        decoded_text = ' '.join(self.dataset.decode(sequences['targets'][0], trim=not verbose))
        print(decoded_text)
        pdb.set_trace()
        to_store = defaultdict(list) # { k: [(combination, translation)] }
        best_bleu = defaultdict(float) # {k: best_bleu }
        ratio = defaultdict(float)
        # for k in range(1, total_num_layer+1):
        use_all_bleu = 0
        count_all = 0.
        count_better = 0.
        for k in range(total_num_layer, 0, -1):
            k_count_all = 0.
            k_count_better = 0.
            for combination in all_combinations[k]:

                # run the data through the model
                sequences = self.translator.translate(batch, combination)
                sequence = next(iter(sequences.values()))[0]
                decoded_text = ' '.join(self.dataset.decode(sequence, trim=not verbose))

                # execute bleu
                bleu = sacrebleu.corpus_bleu([decoded_text], [[target_text]], tokenize="none").score
                #pdb.set_trace()
                if k == 12:
                    use_all_bleu = bleu

                # compare with the best
                # if bleu > best_bleu[k]:
                if bleu > use_all_bleu or k == 12:
                    to_store[k].append((combination.cpu().numpy().tolist(),decoded_text))
                if bleu > use_all_bleu:
                    count_better += 1
                    k_count_better += 1
                count_all += 1
                k_count_all += 1
                    # best_bleu[k] = bleu
                # elif bleu == best_bleu[k]:
                #     to_store[k].append((combination.cpu().numpy().tolist(),decoded_text))

            ratio[k] = k_count_better / k_count_all
            print(datetime.datetime.now())
            print('k: %i best ratio : %.2f' % (k, ratio[k]))

        # store to file
        with open(os.path.join(self.config.output_directory, "example_all_var_%i.json" % batch['example_ids'][0]), 'w') as f:
            obj = {'use_all_bleu': use_all_bleu, 'oracle': to_store, 'better_ratio': count_better / count_all, 'ratio': ratio}
            json.dump(obj, f)


            # for i, example_id in enumerate(batch['example_ids']):
            #     best_k = sorted(best_bleu.items(), key=lambda x: x[1], reverse=True)[0][0]
            #     outputs.append(f'{to_store[best_k][0][1]}\n')
            #     if self.config.order_output:
            #         ordered_outputs.append((example_id, outputs))
            #     else:
            #         output_file.writelines(outputs)

            # for _, outputs in sorted(ordered_outputs, key=lambda x: x[0]): # pylint:disable=consider-using-enumerate
            #     output_file.writelines(outputs)

    def __call__(self, epoch, experiment, verbose=0):
        ''' Generate from the model '''
        enter_mode = experiment.validate
        if self.dataset.split == 'test':
            enter_mode = experiment.test

        with ExitStack() as stack:
            stack.enter_context(enter_mode())
            stack.enter_context(torch.no_grad())

            if not os.path.isdir(self.config.output_directory):
                os.makedirs(self.config.output_directory)

            if self.config.timed:
                Translator.CURRENT = self
                stmt = f'Translator.CURRENT.translate_all(None, {epoch}, None, {verbose})'
                timing = timeit.timeit(stmt, stmt, number=self.config.timed, globals=globals())
                print(f'Translation timing={timing/self.config.timed}')
            else:
                step = experiment.curr_step
                # output_filename = self.config.output_filename or f'translated_{step}.txt'
                # output_path = os.path.join(self.config.output_directory, output_filename)
                # output_file = stack.enter_context(open(output_path, 'wt'))

                # if verbose:
                #     print(f'Outputting to {output_path}')

                # self.translate_all(output_file, epoch, experiment, verbose)
                self.translate_all(None, epoch, experiment, verbose)

                # layermask_path = os.path.join(self.config.output_directory, 'layermasks.txt')
                # with open(layermask_path, 'w') as f:
                #     for lm in self.translator.layermasks:
                #         f.write('\t'.join([str(a.item()) for a in lm]) + '\n')
