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

        if self.config.combination_file is not None:
            with open(self.config.combination_file, 'r') as f:
                self.configs_totranslate = [l.strip() for l in f.readlines()]
        else:
            self.configs_totranslate = None

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

    def translate(self, epoch, combination, verbose=0):

        def get_description():
            description = f'Generate #{epoch}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        batches = tqdm(
            self.dataloader,
            unit='batch',
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout  # needed to make tqdm_wrap_stdout work
        )

        combination_str = ''.join([str(int(c.item())) for c in combination])
        filename = f'{self.config.output_filename}_{combination_str}.txt' if self.config.output_filename is not None else f'oracle_{combination_str}.txt'
        filepath = os.path.join(self.config.output_directory, filename)

        with tqdm_wrap_stdout():
            ordered_outputs = []
            for batch in batches:
                batches.set_description_str(get_description())
                sequences, _ = self.translator.translate(batch, raw_layermask=combination)

                if self.config.timed:
                    continue

                target_sequences = next(iter(sequences.values()))
                for i, example_id in enumerate(batch['example_ids']):
                    sequence = target_sequences[i]
                    decoded = ' '.join(self.dataset.decode(sequence, trim=not verbose))

                    target_text = ' '.join(
                        self.dataset.decode(batch['targets'][i].numpy().tolist(), trim=not verbose))
                    #bleu = sacrebleu.corpus_bleu([decoded], [[target_text]], tokenize="none").score

                    #ordered_outputs.append((example_id, f'{decoded}', bleu))
                    ordered_outputs.append((example_id, f'{decoded}'))

            with open(filepath, 'w') as output_file:
                for _, outputs in sorted(ordered_outputs,
                                         key=lambda x: x[0]):  # pylint:disable=consider-using-enumerate
                    #output_file.write(outputs + "\t" + str(bleu) + "\n")
                    output_file.write(outputs+'\n')

    def translate_all(self, epoch, verbose=0):
        ''' Generate all predictions from the dataset '''
        

        # generate all combinations
        total_num_layer = len(self.modules['model'].encoders) * 2
        all_combinations = self.instantiate_combination(total_num_layer) # {k: [[0,1,0,0...], [1,0,0,0...]]}

        if self.configs_totranslate is not None:
            for c in self.configs_totranslate:
                c = torch.tensor([float(x) for x in c], device=torch.device("cuda"))
                self.translate(epoch, c)
            return

        if self.config.fix_combination is not None:
            fix_combination_tensor = torch.tensor([float(x) for x in self.config.fix_combination])
            len_fcl = fix_combination_tensor.size(0)

        # num bpe tokens
        for k in range(total_num_layer, 0, -1):
            for combination in all_combinations[k]:
                if self.config.fix_combination is not None and torch.sum(combination[:len_fcl] == fix_combination_tensor) != len_fcl:
                    continue
                # if os.path.exists(filepath) and len(open(filepath).readlines()) == 1999:
                #     continue
                print("Generation combination", combination_str)
                self.translate(epoch, combination)


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
