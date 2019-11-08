'''
SynST

--
Main entry point for probing multiheaded attention from SynST


Probe question: does the hard-coded attention actually work when there is a learned head?
need to get the stats of the learned head

'''

from __future__ import print_function

import os
import pdb
import sys
import timeit
from contextlib import ExitStack

import torch
import torch.nn as nn
import pickle
from tqdm import tqdm

from utils import profile
from utils import tqdm_wrap_stdout

from models.utils import MODEL_STATS, STATS_TYPES, probe


class ProbeStatsGetter(object):
    ''' An object that encapsulates model evaluation '''
    CURRENT = None

    def __init__(self, config, model, dataloader, device):

        self.config = config
        self.dataloader = dataloader
        self.translator = model.translator(config).to(device)

        self.modules = {
            'model': model
        }

        self.model = model

        # if 'cuda' in device.type:
        #     self.model = nn.DataParallel(model.cuda())

        self.num_layers = sum(model.decoders[0].enc_dec_attn_config['enc_dec_attn_layer'])
        self.num_heads = max(model.decoders[0].enc_dec_attn_config['enc_dec_attn_num_heads'])
        print("number of enc-dec attn layer ", self.num_layers)

        # stats
        self.train_stats = {model_stat: {stats_type: {'mean': torch.zeros((self.num_layers, self.num_heads),
                                                                    dtype=torch.float32).to(device),
                                                      'var': torch.zeros((self.num_layers, self.num_heads),
                                                                    dtype=torch.float32).to(device)}
                                   for stats_type in STATS_TYPES}
                      for model_stat in MODEL_STATS}
        self.train_count = {model_stat: 0 for model_stat in MODEL_STATS}

        self.test_stats = {model_stat: {stats_type: {'mean': torch.zeros((self.num_layers, self.num_heads),
                                                                         dtype=torch.float32).to(device),
                                                     'var': torch.zeros((self.num_layers, self.num_heads),
                                                                        dtype=torch.float32).to(device)}
                                        for stats_type in STATS_TYPES}
                           for model_stat in MODEL_STATS}
        self.test_count = {model_stat: 0 for model_stat in MODEL_STATS}

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

    def translate_all(self, output_file, stats_file, epoch, experiment, verbose=0):
        ''' Generate all predictions from the dataset '''
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
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )

        with tqdm_wrap_stdout():
            ordered_outputs = []
            with torch.no_grad():
                self.model.eval()
                for batch in batches:
                    # run the data through the model
                    batches.set_description_str(get_description())

                    result = self.model(batch)

                    enc_dec_stats = probe(result['enc_dec_attn_weights_tensor'])
                    train_stats = {'enc_dec_stats': {stats_type: enc_dec_stats[stats_type].view(self.num_layers,
                                                                                                self.num_heads,
                                                                                                -1)
                                                     for stats_type in STATS_TYPES}}

                    self.update_stats(train_stats, self.train_stats, self.train_count)

                    # translate
                    sequences, attn_weights_tensors = self.translator.translate(batch)
                    target_sequences = next(iter(sequences.values()))
                    new_targets = [torch.LongTensor(sequence) for sequence in target_sequences]

                    # new data with new targets
                    self.dataset.collate_field(batch, 'target', new_targets)

                    # model(new-batch)
                    result = self.model(batch)

                    enc_dec_stats = probe(result['enc_dec_attn_weights_tensor'])
                    test_stats = {'enc_dec_stats': {stats_type: enc_dec_stats[stats_type].view(self.num_layers,
                                                                                                self.num_heads,
                                                                                                -1)
                                                     for stats_type in STATS_TYPES}}

                    self.update_stats(test_stats, self.test_stats, self.test_count)

                    if self.config.timed:
                        continue

        self.save_stats(stats_file)

    def update_stats(self, stats, self_stats, self_count):
        ''' Update stats after each batch '''
        for model_stat in stats:
            current_count = stats[model_stat][STATS_TYPES[0]].size()[-1]
            old_count = self_count[model_stat]
            new_count = old_count + current_count
            for stat_type in stats[model_stat]:
                old_mean = self_stats[model_stat][stat_type]['mean']
                current_mean = stats[model_stat][stat_type].mean(dim=-1)
                new_mean = (old_mean * self_count[model_stat] + stats[model_stat][stat_type].sum(dim=-1)) / new_count
                old_var = self_stats[model_stat][stat_type]['var']
                current_var = stats[model_stat][stat_type].var(dim=-1) # torch.sum((stats[model_stat][stat_type] - new_mean.unsqueeze(-1)) ** 2, dim=-1) / (current_count - 1)
                new_var = (old_count * (old_var + (old_mean - new_mean) ** 2)
                           + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
                self_stats[model_stat][stat_type]['mean'] = new_mean
                self_stats[model_stat][stat_type]['var'] = new_var
            self_count[model_stat] = new_count

    # def update_stats(self, stats):
    #     ''' Update stats after each batch '''
    #     for model_stat in stats:
    #         current_count = stats[model_stat][STATS_TYPES[0]].size()[-1]
    #         old_count = self.count[model_stat]
    #         new_count = old_count + current_count
    #         for stat_type in stats[model_stat]:
    #             old_mean = self.stats[model_stat][stat_type]['mean']
    #             current_mean = stats[model_stat][stat_type].sum(dim=-1) / current_count
    #             new_mean = (old_mean * self.count[model_stat] + stats[model_stat][stat_type].sum(dim=-1)) / new_count
    #             old_var = self.stats[model_stat][stat_type]['var']
    #             current_var = torch.sum((stats[model_stat][stat_type] - new_mean.unsqueeze(-1)) ** 2, dim=-1) / (current_count - 1)
    #             new_var = (old_count * (old_var + (old_mean - new_mean) ** 2)
    #                        + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
    #             self.stats[model_stat][stat_type]['mean'] = new_mean
    #             self.stats[model_stat][stat_type]['var'] = new_var
    #         self.count[model_stat] = new_count

    def save_stats(self, stats_file):
        ''' Save stats to file '''
        stats = {'train_stats': self.train_stats, 'train_count': self.train_count,
                 'test_stats': self.test_stats, 'test_count': self.test_count}
        pickle.dump(stats, stats_file, protocol=pickle.HIGHEST_PROTOCOL)

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
                Prober.CURRENT = self
                stmt = f'Translator.CURRENT.translate_all(None, {epoch}, None, {verbose})'
                timing = timeit.timeit(stmt, stmt, number=self.config.timed, globals=globals())
                print(f'Translation timing={timing/self.config.timed}')
            else:
                step = experiment.curr_step
                output_filename = self.config.output_filename or f'translated_{step}.txt'
                output_path = os.path.join(self.config.output_directory, output_filename)
                output_file = stack.enter_context(open(output_path, 'wt'))

                stats_filename = self.config.stats_filename or f'stats_{step}.pickle'
                stats_path = os.path.join(self.config.stats_directory, stats_filename)
                stats_file = stack.enter_context(open(stats_path, 'wb'))

                if verbose:
                    print(f'Outputting to {output_path}')
                    print(f'Stats saving to {stats_path}')

                self.translate_all(output_file, stats_file, epoch, experiment, verbose)

