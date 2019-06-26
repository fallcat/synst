'''
SynST

--
Main entry point for training SynST
'''

from __future__ import print_function

import os
import sys
import time
import shutil
import pickle
from contextlib import ExitStack

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from tqdm import tqdm

import args
import metrics
from actions.evaluate import Evaluator
from data.parallel import chunked_scattering
from models.utils import LinearLRSchedule, WarmupLRSchedule, checkpoint, probe, MODEL_STATS, STATS_TYPES
from utils import profile, tqdm_wrap_stdout, tqdm_unwrap_stdout


class ProbeTrainer(object):
    ''' An object that encapsulates model training '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.stopped_early = False
        self.dataloader = dataloader
        self.validation_dataloader = dataloader
        self.last_checkpoint_time = time.time()

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.optimizer = optim.Adam(model.parameters(), config.base_lr, betas=(0.9, 0.98), eps=1e-9)
        if config.lr_scheduler == 'warmup':
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                WarmupLRSchedule(
                    config.warmup_steps
                )
            )
        elif config.lr_scheduler == 'linear':
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                LinearLRSchedule(
                    config.base_lr,
                    config.final_lr,
                    config.max_steps
                )
            )
        elif config.lr_scheduler == 'exponential':
            self.lr_scheduler = ExponentialLR(
                self.optimizer,
                config.lr_decay
            )
        else:
            raise ValueError('Unknown learning rate scheduler!')

        # Initialize the metrics
        metrics_path = os.path.join(self.config.checkpoint_directory, 'train_metrics.pt')
        self.metric_store = metrics.MetricStore(metrics_path)
        self.metric_store.add(metrics.Metric('oom', metrics.format_int, 't'))
        self.metric_store.add(metrics.Metric('nll', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('lr', metrics.format_scientific, 'g', max_history=1))
        self.metric_store.add(metrics.Metric('num_tok', metrics.format_int, 'a', max_history=1000))

        if self.config.early_stopping:
            self.metric_store.add(metrics.Metric('vnll', metrics.format_float, 'g'))

        self.modules = {
            'model': model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }

        self.num_layers = model.num_layers
        self.num_heads = model.num_heads

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

    def train_epoch(self, epoch, experiment, verbose=0):
        ''' Run one training epoch '''
        oom = self.metric_store['oom']
        learning_rate = self.metric_store['lr']
        num_tokens = self.metric_store['num_tok']
        neg_log_likelihood = self.metric_store['nll']

        def try_optimize(i, last=False):
            # optimize if:
            #  1) last and remainder
            #  2) not last and not remainder
            remainder = bool(i % self.config.accumulate_steps)
            if not last ^ remainder:
                next_lr = self.optimize()

                learning_rate.update(next_lr)
                experiment.log_metric('learning_rate', next_lr)
                return True

            return False

        def get_description():
            description = f'Train #{epoch}'
            if verbose > 0:
                description += f' {self.metric_store}'
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
            i = 1
            nll_per_update = 0.
            length_per_update = 0
            num_tokens_per_update = 0
            for i, batch in enumerate(batches, 1):
                try:
                    nll, length, stats = self.calculate_gradient(batch)
                    self.update_stats(stats)
                    did_optimize = try_optimize(i)

                    # record the effective number of tokens
                    num_tokens_per_update += int(sum(batch['input_lens']))
                    num_tokens_per_update += int(sum(batch['target_lens']))

                    if length:
                        # record length and nll
                        nll_per_update += nll
                        length_per_update += length

                    if did_optimize:
                        # advance the experiment step
                        experiment.set_step(experiment.curr_step + 1)

                        num_tokens.update(num_tokens_per_update)
                        neg_log_likelihood.update(nll_per_update / length_per_update)

                        experiment.log_metric('num_tokens', num_tokens_per_update)
                        experiment.log_metric('nll', neg_log_likelihood.last_value)

                        nll_per_update = 0.
                        length_per_update = 0
                        num_tokens_per_update = 0

                except RuntimeError as rte:
                    if 'out of memory' in str(rte):
                        torch.cuda.empty_cache()

                        oom.update(1)
                        experiment.log_metric('oom', oom.total)
                    else:
                        batches.close()
                        raise rte

                if self.should_checkpoint():
                    new_best = False
                    if self.config.early_stopping:
                        with tqdm_unwrap_stdout():
                            new_best = self.evaluate(experiment, epoch, verbose)

                    self.checkpoint(epoch, experiment.curr_step, new_best)

                batches.set_description_str(get_description())
                if self.is_done(experiment, epoch):
                    batches.close()
                    break

            try_optimize(i, last=True)

    def update_stats(self, stats):
        ''' Update stats after each batch '''
        for model_stat in stats:
            current_count = stats[model_stat][STATS_TYPES[0]].size()[-1]
            old_count = self.count[model_stat]
            new_count = old_count + current_count
            for stat_type in stats[model_stat]:
                old_mean = self.stats[model_stat][stat_type]['mean']
                current_mean = stats[model_stat][stat_type].mean(dim=-1)
                new_mean = (old_mean * self.count[model_stat] + stats[model_stat][stat_type].sum(dim=-1)) / new_count
                old_var = self.stats[model_stat][stat_type]['var']
                current_var = stats[model_stat][stat_type].var(dim=-1) # torch.sum((stats[model_stat][stat_type] - new_mean.unsqueeze(-1)) ** 2, dim=-1) / (current_count - 1)
                new_var = (old_count * (old_var + (old_mean - new_mean) ** 2)
                           + current_count * (current_var + (current_mean - new_mean) ** 2)) / new_count
                self.stats[model_stat][stat_type]['mean'] = new_mean
                self.stats[model_stat][stat_type]['var'] = new_var
            self.count[model_stat] = new_count

    def save_stats(self, stats_file):
        ''' Save stats to file '''
        stats = {'stats': self.stats, 'count': self.count}
        pickle.dump(stats, stats_file, protocol=pickle.HIGHEST_PROTOCOL)

    def should_checkpoint(self):
        ''' Function which determines if a new checkpoint should be saved '''
        return time.time() - self.last_checkpoint_time > self.config.checkpoint_interval

    def checkpoint(self, epoch, step, best=False):
        ''' Save a checkpoint '''
        checkpoint_path = checkpoint(
            epoch, step, self.modules,
            self.config.checkpoint_directory,
            max_checkpoints=self.config.max_checkpoints
        )

        if best:
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
            shutil.copy2(checkpoint_path, best_checkpoint_path)

        self.metric_store.save()
        self.last_checkpoint_time = time.time()

    def evaluate(self, experiment, epoch, verbose=0):
        ''' Evaluate the current model and determine if it is a new best '''
        model = self.modules['model']
        evaluator = Evaluator(args.ArgGroup(None), model, self.validation_dataloader, self.device)
        vnll = evaluator(epoch, experiment, verbose)
        metric = self.metric_store['vnll']
        full_history = metric.values
        metric.update(vnll)
        self.metric_store.save()

        return all(vnll < nll for nll in full_history[:-1])

    def is_done(self, experiment, epoch):
        ''' Has training completed '''
        if self.config.max_steps and experiment.curr_step >= self.config.max_steps:
            return True

        if self.config.max_epochs and epoch >= self.config.max_epochs:
            return True

        if self.config.early_stopping:
            history = self.metric_store['vnll'].values[-self.config.early_stopping - 1:]
            if len(history) == self.config.early_stopping + 1:
                self.stopped_early = all(history[-1] > nll for nll in history[:-1])
                return self.stopped_early

        return False

    def optimize(self):
        ''' Calculate an optimization step '''
        self.lr_scheduler.step()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.lr_scheduler.get_lr()[0]

    def calculate_gradient(self, batch):
        ''' Runs one step of optimization '''
        # run the data through the model
        self.model.train()
        result = self.model(batch)
        loss = result['smoothed_nll']
        nll = result['nll']

        # stats
        encoder_stats = probe(result['encoder_attn_weights_tensor'])
        decoder_stats = probe(result['decoder_attn_weights_tensor'])
        enc_dec_stats = probe(result['enc_dec_attn_weights_tensor'])
        stats = {'encoder_stats': {stats_type: encoder_stats[stats_type].view(self.num_layers,
                                                                              self.num_heads,
                                                                              -1)
                           for stats_type in STATS_TYPES},
                 'decoder_stats': {stats_type: decoder_stats[stats_type].view(self.num_layers,
                                                                              self.num_heads,
                                                                              -1)
                           for stats_type in STATS_TYPES},
                 'enc_dec_stats': {stats_type: enc_dec_stats[stats_type].view(self.num_layers,
                                                                              self.num_heads,
                                                                              -1)
                           for stats_type in STATS_TYPES}}

        # nn.DataParallel wants to gather rather than doing a reduce_add, so the output here
        # will be a tensor of values that must be summed
        nll = nll.sum()
        loss = loss.sum()

        # calculate gradients then run an optimization step
        loss.backward()

        # need to use .item() which converts to Python scalar
        # because as a Tensor it accumulates gradients
        return nll.item(), torch.sum(batch['target_lens']).item(), stats

    def __call__(self, start_epoch, experiment, verbose=0):
        ''' Execute training '''
        with ExitStack() as stack:
            stack.enter_context(chunked_scattering())
            stack.enter_context(experiment.train())

            if start_epoch > 0 or experiment.curr_step > 0:
                self.metric_store = self.metric_store.load()

            epoch = start_epoch
            experiment.log_current_epoch(epoch)

            stats_filename = self.config.stats_filename or f'train_stats.pickle'
            stats_path = os.path.join(self.config.stats_directory, stats_filename)
            stats_file = stack.enter_context(open(stats_path, 'wb'))

            while not self.is_done(experiment, epoch):
                experiment.log_current_epoch(epoch)
                self.train_epoch(epoch, experiment, verbose)
                experiment.log_epoch_end(epoch)
                epoch += 1

            self.save_stats(stats_file)

            if self.stopped_early:
                print('Stopping early!')
            else:
                new_best = False
                if self.config.early_stopping:
                    new_best = self.evaluate(experiment, epoch, verbose)

                self.checkpoint(epoch, experiment.curr_step, new_best)
