'''
SynST

--
Main entry point for evaluating SynST
'''

from __future__ import print_function

import os
import sys
import signal
import time
import pickle
import atexit
from contextlib import ExitStack

import torch
from torch import nn
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import metrics
from models.utils import restore, probe, MODEL_STATS, STATS_TYPES
from utils import profile
from utils import tqdm_wrap_stdout


class CheckpointEventHandler(FileSystemEventHandler):
    ''' A filesystem event handler for new checkpoints '''
    def __init__(self, handler, experiment, verbose=0):
        ''' Initialize the CheckpointEventHandler '''
        super(CheckpointEventHandler, self).__init__()
        self.watches = set()
        self.handler = handler
        self.verbose = verbose
        self.experiment = experiment

    def on_created(self, event):
        ''' Watcher for a new file '''
        root, ext = os.path.splitext(event.src_path)
        basename = os.path.basename(root)
        if ext == '.incomplete' and basename == 'checkpoint.pt':
            self.watches.add(event.src_path)

            if self.verbose > 1:
                print(f'Waiting for {event.src_path}')

    def on_moved(self, event):
        ''' Handle when a file has been modified '''
        if event.src_path in self.watches:
            self.watches.remove(event.src_path)
            self.handler(event.dest_path, self.experiment, self.verbose)


class ProbeEvaluator(object):
    ''' An object that encapsulates model evaluation '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.dataloader = dataloader

        self.should_exit = False
        signal.signal(signal.SIGHUP, self.on_training_complete)

        self.observer = None

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.modules = {
            'model': model
        }

        # stats
        self.stats = {model_stat: {stats_type: {'mean': torch.zeros((model.num_layers, model.num_heads),
                                                                    dtype=torch.float32).to(device),
                                                'var': torch.zeros((model.num_layers, model.num_heads),
                                                                   dtype=torch.float32).to(device)}
                                   for stats_type in STATS_TYPES}
                      for model_stat in MODEL_STATS}
        self.count = {model_stat: 0 for model_stat in MODEL_STATS}

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    def evaluate(self, batch):
        ''' Runs one evaluation step '''
        with torch.no_grad():
            self.model.eval()
            # _, nll = self.model(batch)
            result = self.model(batch)
            nll = result['nll']

            # stats
            encoder_stats = probe(result['encoder_attn_weights_tensor'])
            decoder_stats = probe(result['decoder_attn_weights_tensor'])
            enc_dec_stats = probe(result['enc_dec_attn_weights_tensor'])
            stats = {'encoder_stats': {stats_type: encoder_stats[stats_type].view(self.model.num_layers,
                                                                                  self.model.num_heads,
                                                                                  -1)
                                       for stats_type in STATS_TYPES},
                     'decoder_stats': {stats_type: decoder_stats[stats_type].view(self.model.num_layers,
                                                                                  self.model.num_heads,
                                                                                  -1)
                                       for stats_type in STATS_TYPES},
                     'enc_dec_stats': {stats_type: enc_dec_stats[stats_type].view(self.model.num_layers,
                                                                                  self.model.num_heads,
                                                                                  -1)
                                       for stats_type in STATS_TYPES}}

            # nn.DataParallel wants to gather rather than doing a reduce_add, so the output here
            # will be a tensor of values that must be summed
            nll = nll.sum()

            # need to use .item() which converts to Python scalar
            # because as a Tensor it accumulates gradients
            return nll.item(), torch.sum(batch['target_lens']).item(), stats


    def evaluate_epoch(self, epoch, experiment, stats_file, verbose=0):
        ''' Evaluate a single epoch '''
        neg_log_likelihood = metrics.Metric('nll', metrics.format_float)

        def get_description():
            mode_name = 'Test' if self.dataset.split == 'test' else 'Validate'
            description = f'{mode_name} #{epoch}'
            if verbose > 0:
                description += f' {neg_log_likelihood}'
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
            for batch in batches:
                # run the data through the model
                batches.set_description_str(get_description())
                nll, length, stats = self.evaluate(batch)
                self.update_stats(stats, self.stats, self.count)
                if length:
                    neg_log_likelihood.update(nll / length)

        experiment.log_metric('nll', neg_log_likelihood.average)
        self.save_stats(stats_file)
        return neg_log_likelihood.average

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
                current_var = stats[model_stat][stat_type].var(
                    dim=-1)  # torch.sum((stats[model_stat][stat_type] - new_mean.unsqueeze(-1)) ** 2, dim=-1) / (current_count - 1)
                new_var = (old_count * (old_var + (old_mean - new_mean) ** 2)
                           + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
                self_stats[model_stat][stat_type]['mean'] = new_mean
                self_stats[model_stat][stat_type]['var'] = new_var
            self_count[model_stat] = new_count

    def save_stats(self, stats_file):
        ''' Save stats to file '''
        stats = {'stats': self.stats, 'count': self.count}
        pickle.dump(stats, stats_file, protocol=pickle.HIGHEST_PROTOCOL)

    def on_new_checkpoint(self, path, experiment, verbose=0):
        ''' Upon receiving a new checkpoint path '''
        epoch, step = restore(
            path,
            self.modules,
            num_checkpoints=self.config.average_checkpoints,
            map_location=self.device.type
        )
        experiment.set_step(step)
        self.evaluate_epoch(epoch, experiment, verbose)

    def on_training_complete(self, signum, frame): # pylint:disable=unused-argument
        ''' Received a SIGHUP indicating the training session has ended '''
        self.should_exit = True

    def shutdown(self):
        ''' Shutdown the current observer '''
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def watch(self, experiment, verbose=0):
        ''' Watch for a new checkpoint and run an evaluation step '''
        # Use a polling observer because slurm doesn't seem to correctly handle inotify events :/
        self.observer = PollingObserver() if self.config.polling else Observer()
        event_handler = CheckpointEventHandler(self.on_new_checkpoint, experiment, verbose)
        self.observer.schedule(event_handler, path=self.config.watch_directory)
        self.observer.start()

        while not self.should_exit:
            time.sleep(1)

        atexit.register(self.shutdown)

    def __call__(self, epoch, experiment, verbose=0):
        ''' Validate the model and store off the stats '''
        enter_mode = experiment.validate
        if self.dataset.split == 'test':
            enter_mode = experiment.test

        with ExitStack() as stack:
            stack.enter_context(enter_mode())
            stack.enter_context(torch.no_grad())

            stats_filename = self.config.stats_filename or f'stats_probe_evaluate.pickle'
            stats_path = os.path.join(self.config.stats_directory, stats_filename)
            stats_file = stack.enter_context(open(stats_path, 'wb'))

            if self.config.watch_directory:
                self.watch(experiment, verbose)
            else:
                return self.evaluate_epoch(epoch, experiment, stats_file, verbose)
