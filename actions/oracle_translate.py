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


class OracleTranslator(object):
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

    def translate_all(self, epoch, verbose=0):
        ''' Generate all predictions from the dataset '''
        def get_description():
            description = f'Generate #{epoch}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        # generate all combinations
        total_num_layer = len(self.modules['model'].encoders) * 2
        all_combinations = self.instantiate_combination(total_num_layer) # {k: [[0,1,0,0...], [1,0,0,0...]]}

        if self.config.fix_combination is not None:
            fix_combination_list = [int(x) for x in self.config.fix_combination]
            len_fcl = len(fix_combination_list)

        # num bpe tokens
        for k in range(total_num_layer, 0, -1):
            for combination in all_combinations[k]:
                if self.config.fix_combination is not None and combination[:len_fcl] != fix_combination_list:
                    continue
                combination_str = ''.join([str(int(c.item())) for c in combination])
                print("Generation combination", combination_str)

                batches = tqdm(
                    self.dataloader,
                    unit='batch',
                    dynamic_ncols=True,
                    desc=get_description(),
                    file=sys.stdout  # needed to make tqdm_wrap_stdout work
                )

                with tqdm_wrap_stdout():
                    ordered_outputs = []
                    for batch in batches:
                        batches.set_description_str(get_description())
                        sequences, _ = self.translator.translate(batch, raw_layermask=combination)

                        if self.config.timed:
                            continue

                        target_sequences = next(iter(sequences.values()))
                        for i, example_id in enumerate(batch['example_ids']):
                            outputs = []
                            sequence = target_sequences[i]
                            decoded = ' '.join(self.dataset.decode(sequence, trim=not verbose))
                            outputs.append(f'{decoded}\n')

                            target_text = ' '.join(
                                self.dataset.decode(batch['targets'][i].numpy().tolist(), trim=not verbose))
                            bleu = sacrebleu.corpus_bleu([decoded], [[target_text]], tokenize="none").score

                            ordered_outputs.append((example_id, outputs, bleu))

                    filename = f'{self.config.output_filename}_{combination_str}.txt' or f'oracle_{combination_str}.txt'
                    with open(os.path.join(self.config.output_directory, filename), 'w') as output_file:
                        for _, outputs, bleu in sorted(ordered_outputs,
                                                 key=lambda x: x[0]):  # pylint:disable=consider-using-enumerate
                            output_file.writelines(outputs + "\t" + str(bleu))

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
                OracleTranslator.CURRENT = self
                stmt = f'Translator.CURRENT.translate_all(None, {epoch}, None, {verbose})'
                timing = timeit.timeit(stmt, stmt, number=self.config.timed, globals=globals())
                print(f'Translation timing={timing/self.config.timed}')
            else:
                step = experiment.curr_step
                self.translate_all(epoch, verbose)
