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
from models.utils import LinearLRSchedule, WarmupLRSchedule, WarmupLRSchedule2, DummyLRSchedule, checkpoint, Translator
from utils import profile, tqdm_wrap_stdout, tqdm_unwrap_stdout
import datetime

import sacrebleu
import numpy as np
from itertools import combinations
import random
import pdb
import logging

class IterativeTrainer(object):
    ''' An object that encapsulates model training '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.stopped_early = False
        self.dataloader = dataloader
        self.validation_dataloader = dataloader
        self.validation_dataset = dataloader.dataset
        self.last_checkpoint_time = time.time()

        if self.config.freeze_layermask:
            print('freeze layermask predictor')
            model.layer_mask_predictor.projection.weight.requires_grad = False
            model.layer_mask_predictor.projection.bias.requires_grad = False

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        params = []
        for pname, pval in model.named_parameters():
            if 'layer_mask_predictor' in pname:
                pval.requires_grad = False
                continue
            params.append(pval)

        if self.config.optimizer == "adam":
            # self.optimizer = optim.Adam(model.parameters(), config.base_lr, betas=(0.9, 0.98), eps=1e-9)
            self.optimizer = optim.Adam(params, config.base_lr, betas=(0.9, 0.98), eps=1e-9)
            if config.lr_scheduler == 'warmup':
                self.lr_scheduler = LambdaLR(
                    self.optimizer,
                    WarmupLRSchedule(
                        config.warmup_steps
                    )
                )

            elif config.lr_scheduler == 'warmup2':
                self.lr_scheduler = LambdaLR(
                    self.optimizer,
                    WarmupLRSchedule2(
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

        elif self.config.optimizer == "sgd":
            print("using optimizer: SGD")
            self.optimizer = optim.SGD(model.parameters(), lr=config.base_lr, momentum=0.9)
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                DummyLRSchedule(
                    config.base_lr
                )
            )

        elif self.config.optimizer == "adam-fixed":
            print("using optimizer: adam with fixed learning rate")
            self.optimizer = optim.Adam(model.parameters(), config.base_lr, betas=(0.9, 0.98), eps=1e-9)
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                DummyLRSchedule(
                    config.base_lr
                )
            )

        else:
            raise ValueError('Unknown optimizer!') 


        # Initialize the metrics
        metrics_path = os.path.join(self.config.checkpoint_directory, 'train_metrics.pt')
        self.metric_store = metrics.MetricStore(metrics_path)
        self.metric_store.add(metrics.Metric('oom', metrics.format_int, 't'))
        self.metric_store.add(metrics.Metric('nll', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('lr', metrics.format_scientific, 'g', max_history=1))
        self.metric_store.add(metrics.Metric('num_tok', metrics.format_int, 'a', max_history=1000))
        self.metric_store.add(metrics.Metric('reward', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('layermask', metrics.format_float, 'g', max_history=1))
        self.metric_store.add(metrics.Metric('test_bleu', metrics.format_float, 'g', max_history=1000))
        self.metric_store.add(metrics.Metric('combined_test_bleu', metrics.format_float, 'g', max_history=1000))
        self.metric_store.add(metrics.Metric('percent_ge', metrics.format_float, 'g', max_history=1000))
        # self.metric_store.add(metrics.Metric('time_per_batch', metrics.format_float, 'g', max_history=100000))
        # self.metric_store.add(metrics.Metric('time_total', metrics.format_float, 'g', max_history=1))

        if self.config.early_stopping:
            self.metric_store.add(metrics.Metric('vnll', metrics.format_float, 'g'))

        self.modules = {
            'model': model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }


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
        rl_reward = self.metric_store['reward']
        log_layermask = self.metric_store['layermask']

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
            desc="",#get_description(),
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )

        with tqdm_wrap_stdout():
            if self.config.debug: # only for debug
                self.oracle_sample_experiment(experiment)
                return

        with tqdm_wrap_stdout():
            i = 1
            nll_per_update = 0.
            length_per_update = 0
            num_tokens_per_update = 0
            reward_per_update = 0.
            reward_num = 0
            for i, batch in enumerate(batches, 1):

                # optimize
                try:
                    
                    nll, length, reward, sum_layermask = self.calculate_gradient(batch, experiment.curr_step)
                    did_optimize = try_optimize(i)

                    # record the effective number of tokens
                    num_tokens_per_update += int(sum(batch['input_lens']))
                    num_tokens_per_update += int(sum(batch['target_lens']))

                    if length:
                        # record length and nll
                        nll_per_update += nll
                        length_per_update += length
                        reward_per_update += reward
                        reward_num += 1

                    if did_optimize:
                        # advance the experiment step
                        experiment.set_step(experiment.curr_step + 1)

                        num_tokens.update(num_tokens_per_update)
                        neg_log_likelihood.update(nll_per_update / length_per_update)
                        rl_reward.update(reward_per_update / reward_num)
                        log_layermask.update(sum_layermask)

                        experiment.log_metric('num_tokens', num_tokens_per_update)
                        experiment.log_metric('nll', neg_log_likelihood.last_value)
                        experiment.log_metric('reward', rl_reward.last_value)
                        experiment.log_metric('layermask', log_layermask.last_value)

                        # experiment.log_metric('max_memory_alloc', torch.cuda.max_memory_allocated()//1024//1024)
                        # experiment.log_metric('max_memory_cache', torch.cuda.max_memory_cached()//1024//1024)

                        nll_per_update = 0.
                        length_per_update = 0
                        num_tokens_per_update = 0
                        reward_per_update = 0.
                        reward_num = 0

                except RuntimeError as rte:
                    if 'out of memory' in str(rte):
                        torch.cuda.empty_cache()

                        oom.update(1)
                        experiment.log_metric('oom', oom.total)
                        #exit(-1)
                        raise rte
                    else:
                        batches.close()
                        raise rte

                if self.should_checkpoint():
                    new_best = False
                    if self.config.early_stopping:
                        with tqdm_unwrap_stdout():
                            new_best = self.evaluate(experiment, epoch, verbose)
                    self.checkpoint(epoch, experiment.curr_step, new_best)

                # sample layermask
                # if experiment.curr_step >= self.config.step_start_iter_train and experiment.curr_step % self.config.sample_interval == 0 and did_optimize:
                #     self.update_layermask_predictor(experiment.curr_step)

                batches.set_description_str(get_description())
                if self.is_done(experiment, epoch):
                    batches.close()
                    break

            try_optimize(i, last=True)

    def get_translated(self, translator, batches, gold=False, layermask=None):

        translated = []
        masks = []
        for bi, b in enumerate(batches):
            t, m = translator.translate(b, raw_layermask=layermask)
            translated.append(t)
            masks.append(m)

        eval_batch_gen = []
        eval_batch_gold = []
        # eval all-on
        for t in translated:
            for i in range(len(t['targets'])):
                decoded = ' '.join(self.validation_dataset.decode(t['gold_targets'][i], trim=True))
                eval_batch_gold.append(decoded)  
                decoded = ' '.join(self.validation_dataset.decode(t['targets'][i], trim=True))
                eval_batch_gen.append(decoded)  

        return eval_batch_gold, eval_batch_gen, masks   

    def oracle_sample_experiment(self, experiment):
        
        print("oracle experiment: configs per iter: {} lr:{} n_batches:{} lmp_train_steps:{}".format(self.config.iter_train_n_configs, self.config.base_lr, self.config.iter_train_lmp_val_n_batches, self.config.max_lmp_train_steps))

        # train on validation and test on test
        model = self.modules['model']
        # set dropout to 0
        model.eval() # dropout to 0 and no_grad
    
        ssize, s_bsize = self.config.sample_size, self.config.sample_batch_size
        test_dataloader = iter(self.validation_dataloader)  # this is actually test during debugging mode, need to specify total num sentences in the script
        valid_dataloader = iter(self.dataloader)

        # test_dataloader, get all-on bleu by setences
        iter_times = ssize // s_bsize
        test_batches = [next(test_dataloader) for i in range(iter_times)]
        valid_batches = [next(valid_dataloader) for i in range(iter_times)]

        model.set_LMP_type('noskip')
        noskip_translator = model.translator(self.config).to(torch.device("cuda"))
        test_batch_gold, test_allon_batch_gen, _ = self.get_translated(noskip_translator, test_batches, gold=True)
        #assert len(test_allon_batch_gen) == 1999
        all_on_bleu = sacrebleu.corpus_bleu(test_allon_batch_gen, [test_batch_gold], tokenize='none').score
        test_allon_bleu_by_sent = [sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score for gen_i, gold_i in zip(test_allon_batch_gen, test_batch_gold)]
        
        val_batch_gold, val_allon_batch_gen, _ = self.get_translated(noskip_translator, valid_batches, gold=True)
        all_on_val_bleu = sacrebleu.corpus_bleu(val_allon_batch_gen, [val_batch_gold], tokenize='none').score
        val_allon_bleu_by_sent = [sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score for gen_i, gold_i in zip(val_allon_batch_gen, val_batch_gold)]

        print("all-on test corpus bleu {}\n".format(all_on_bleu))
        print("all-on val corpus bleu {}\n".format(all_on_val_bleu))

        model.set_LMP_type('iterative_training_debug_oracle')
        # train
        # all combinations
        num_layer = 2 * len(model.encoders)
        # generate all combinations, update: also included the all-on config
        all_combs = sum([list(combinations(range(num_layer), k)) for k in range(1, num_layer+1)], [])
        # filter those without decoder
        all_combs = [x for x in all_combs if any(y >= num_layer//2 for y in x)]
        random.Random(42).shuffle(all_combs)
        # get the index of all-on config
        ci_allon = all_combs.index(tuple(i for i in range(num_layer)))
        print("all-on config ci: %i" % ci_allon)

        sample_translator = model.translator(self.config).to(torch.device("cuda"))
        j_start, j_size = 0, self.config.iter_train_n_configs

        # group valid batches
        n = self.config.iter_train_lmp_val_n_batches
        valid_batches = [(valid_batches[i:i+n]) for i in range(0, iter_times, n)]
        for gi, group_val_batches in enumerate(valid_batches): # one group is responsible for optimizing j_size configs
            # use (gi*n-1)th batch as the validation set for lmp training
            lmp_val_batch = valid_batches[-1][-1]
            # get "ground truth"
            agg_stats_list = []
            for i, val_batch in enumerate(group_val_batches):
                aggregate_stats = torch.zeros(s_bsize, len(all_combs), device=torch.device("cuda"))
                # get "ground truth" for these j_size configs
                for ci, comb in enumerate(all_combs[j_start * j_size : (j_start+1) * j_size]):
                    layermask = torch.zeros(s_bsize, num_layer, device=torch.device("cuda")) 
                    for activate_i in comb:
                        layermask[:, activate_i] += 1 # now the whole batch use the same mask
                    if len(val_batch['inputs']) % s_bsize != 0:
                        layermask = layermask[:-1, :]
                    _, batch_gen, _ = self.get_translated(sample_translator, [val_batch], layermask=layermask)
                    for eval_i, (gold_i, gen_i) in enumerate(zip(val_batch_gold[(n*gi+i)*s_bsize : (n*gi+i+1)*s_bsize], batch_gen)):
                        this_bleu = sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score
                        all_on_bleu = sacrebleu.corpus_bleu([val_allon_batch_gen[eval_i+(n*gi+i)*s_bsize]], [[gold_i]], tokenize='none').score
                        try:
                            assert all_on_bleu == val_allon_bleu_by_sent[eval_i+(n*gi+i)*s_bsize]
                        except:
                            pdb.set_trace()
                        if this_bleu > val_allon_bleu_by_sent[eval_i+(n*gi+i)*s_bsize]:
                            aggregate_stats[eval_i, ci + j_start * j_size] += 1
                            # print("batch {} example {} config {}: this bleu {:.2f} all-on bleu {:.2f}" .format(i, eval_i, ci, this_bleu, all_on_bleu))
                # set all all-on config to 1
                aggregate_stats[:, ci_allon] = 1
                if len(val_batch['inputs']) % s_bsize != 0:
                    neg = aggregate_stats[:-1, j_start * j_size : (j_start+1) * j_size]
                else:
                    neg = aggregate_stats[:, j_start * j_size : (j_start+1) * j_size]
                neg[neg == 0] = -1
                agg_stats_list.append(aggregate_stats)
            self.enable_train_LMP(model)
            lmp_optimizer = optim.Adam(model.layer_mask_predictor.parameters(), lr=self.config.base_lr, betas=(0.9, 0.98), eps=1e-9)
            b_flag = False
            prev_percent = -sys.maxsize
            for i in range(self.config.max_lmp_train_steps):
                for zi, (batch, stats) in enumerate(zip(group_val_batches, agg_stats_list)):
                    embedding = model.embed(batch['inputs'].cuda(), model.embedding)
                    padding_masks = batch['inputs'].eq(model.padding_idx).cuda()
                    loss, _ = model.layer_mask_predictor(embedding, padding_masks, aggregate_stats=stats) # stats after the second iteration should be 30~60 0but still 0-30
                    loss.backward()
                    lmp_optimizer.step()
                    lmp_optimizer.zero_grad()
                    if i % 200 == 0:
                        # validation + early stopping
                        self.disable_train_LMP(model)
                        model.set_LMP_config_range(0, (j_start+1) * j_size)
                        lmp_val_batch_gold, lmp_val_batch_gen, lmp_val_batch_masks  = self.get_translated(sample_translator, [lmp_val_batch])
                        lmp_val_allon_batch_gen = val_allon_batch_gen[-s_bsize:]
                        # get percent_G
                        lmp_gen_bleu = [sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score for gen_i, gold_i in zip(lmp_val_batch_gen, lmp_val_batch_gold)]
                        lmp_allon_bleu = [sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score for gen_i, gold_i in zip(lmp_val_allon_batch_gen, lmp_val_batch_gold)]
                        this_percent = sum([int(a >= b) for a, b in zip(lmp_gen_bleu, lmp_allon_bleu)]) / len(lmp_val_batch_gold)
                        print('{} {} {}'.format(i, loss.item(), this_percent))
                        # compare the logged validation percentage 
                        if prev_percent > this_percent or prev_percent >= 0.9:
                            b_flag = True
                            break
                        else:
                            self.enable_train_LMP(model)
                            if zi == len(group_val_batches) - 1 and this_percent >= prev_percent:
                                prev_percent = this_percent
                if b_flag:
                    break
            self.disable_train_LMP(model)
            # test LMP on test set
            # only test the configs that optimized
            model.set_LMP_config_range(0, (j_start+1) * j_size)
            _, test_batch_gen, test_batch_masks = self.get_translated(sample_translator, test_batches)
            test_batch_bleu_by_sent = [sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score for gen_i, gold_i in zip(test_batch_gen, test_batch_gold)]
            test_batch_bleu = sacrebleu.corpus_bleu(test_batch_gen, [test_batch_gold], tokenize='none').score
            print("==========lmp using 0 ~ {} configs===========".format((j_start+1) * j_size))
            print("test corpus bleu: {:.2f}".format(test_batch_bleu))
            percent_g = sum([int(a >= b) for a, b in zip(test_batch_bleu_by_sent, test_allon_bleu_by_sent)]) / len(test_allon_batch_gen)
            print("percent >= : {}".format(percent_g))
            combined_test_gen = [test_batch_gen[i] if test_batch_bleu_by_sent[i] > test_allon_bleu_by_sent[i] else test_allon_batch_gen[i] for i in range(len(test_batch_gen))]
            combined_test_bleu = sacrebleu.corpus_bleu(combined_test_gen, [test_batch_gold], tokenize='none').score
            print("combined test corpus bleu: {:.2f}".format(combined_test_bleu))
            ratio = sum([a.sum(dim=0) for a in test_batch_masks])/(len(test_batch_masks) * test_batch_masks[0].shape[0])
            print("layer selection ratio: {}".format(np.around(ratio.cpu().numpy(), 2).tolist()))
            all_on_masks = sum([(m.sum(dim=1) == num_layer).sum() for m in test_batch_masks])
            print("all-on ratio: {}".format(all_on_masks.item() / (s_bsize * len(test_batch_masks)))) # what percent of the test set selecting all-on config
            experiment.log_metric("test_bleu", test_batch_bleu)
            experiment.log_metric("combined_test_bleu", combined_test_bleu)
            experiment.log_metric("percent_ge", percent_g)
            if not self.config.optimize_the_same:
                j_start += 1

        # store generated
        with open(os.path.join(self.config.checkpoint_directory, 'lmp_only_translated.txt'), 'w') as f:
            for line in test_batch_gen:
                f.write(line + '\n')

        with open(os.path.join(self.config.checkpoint_directory, 'lmp_combined_translated.txt'), 'w') as f:
            for line in combined_test_gen:
                f.write(line + '\n')


    def enable_train_LMP(self, model):
        for pname, pval in model.named_parameters():
            if 'layer_mask_predictor' in pname:
                pval.requires_grad = True
                continue

    def disable_train_LMP(self, model):
        for pname, pval in model.named_parameters():
            if 'layer_mask_predictor' in pname:
                pval.requires_grad = True
                continue

    def update_layermask_predictor(self, curr_step):

        # sample from validation data (validation dataloader, shuffle=True, everytime only inference on first 300)
        model = self.modules['model']
        # set dropout to 0
        model.eval() # dropout to 0 and no_grad
        
        
        ssize, s_bsize = self.config.sample_size, self.config.sample_batch_size
        data_loader = iter(self.validation_dataloader) 
        
        iter_times = ssize // s_bsize
        batch = [next(data_loader) for i in range(iter_times)]

        model.set_LMP_type('noskip')
        noskip_translator = model.translator(self.config).to(torch.device("cuda"))
        translated = [noskip_translator.translate(b) for b in batch]
        eval_batch_gold, eval_batch_gen = [], []

        # eval all-on
        for t in translated:
            for i in range(len(t['targets'])):
                decoded = ' '.join(self.validation_dataset.decode(t['targets'][i], trim=True))
                gold = ' '.join(self.validation_dataset.decode(t['gold_targets'][i], trim=True))
                eval_batch_gen.append(decoded)           
                eval_batch_gold.append(gold)
        
        all_on_bleu = sacrebleu.corpus_bleu(eval_batch_gen, [eval_batch_gold], tokenize='none').score
        print("all-on bleu %f" % all_on_bleu)
        model.set_LMP_type('iterative_training')
        # sample multiple times and eval
        # init as zero, and add model.layermask each time to count frequency
        write_fname = os.path.join(self.config.checkpoint_directory, 'lmp_train_%i.txt' % (curr_step))
        with open(write_fname, 'w') as f:
            f.write("Start sampling : %s\n" % datetime.datetime.now())

        # all combinations
        num_layer = 2 * len(model.encoders)
        # generate all combinations
        all_combs = sum([list(combinations(range(num_layer), k)) for k in range(1, num_layer)], [])
        # filter those without decoder
        all_combs = [x for x in all_combs if any(y >= num_layer//2 for y in x)]

        masks = torch.zeros(ssize, 2 * len(model.encoders), device=torch.device("cuda")) # [bs, #layers]
        for st in range(self.config.sample_times):
            # sample layermask
            eval_batch_gen = []
            layermask = torch.zeros_like(masks) 
            chosen_comb = random.choices(all_combs, k=layermask.shape[0])
            for i, comb in enumerate(chosen_comb):
                for j in comb:
                    layermask[i, j] += 1
            sample_translator = model.translator(self.config).to(torch.device("cuda"))
            translated = [sample_translator.translate(b, raw_layermask=layermask[i*s_bsize:(i+1)*s_bsize]) for i, b in enumerate(batch)]
            for t in translated:
                for i in range(len(t['targets'])):
                    decoded = ' '.join(self.validation_dataset.decode(t['targets'][i], trim=True))
                    eval_batch_gen.append(decoded)
            
            this_bleu = sacrebleu.corpus_bleu(eval_batch_gen, [eval_batch_gold], tokenize='none').score
            print("sample %i : this bleu %f" % (st, this_bleu))

            if this_bleu >= all_on_bleu:
                print(layermask)
                masks += layermask

        with open(write_fname, 'a+') as f:
            f.write("end sampling %i at %s\n" % (st, datetime.datetime.now()))

        # new distribution
        new_dist = masks / self.config.sample_times
        print(new_dist)
        # train LMP with new_distribution
        # optimizer, only optimize 
        self.enable_train_LMP(model)
        lmp_optimizer = optim.Adam(model.layer_mask_predictor.parameters(), lr=3e-3, betas=(0.9, 0.98), eps=1e-9)
        
        loss_curve = []
        for i in range(self.config.max_lmp_train_steps // iter_times):
            avg_loss = 0
            for bi, b in enumerate(batch):
                embedding = model.embed(b['inputs'].cuda(), model.embedding)
                padding_masks = b['inputs'].eq(model.padding_idx).cuda()
                loss, _ = model.layer_mask_predictor(embedding, padding_masks, aggregate_stats=new_dist[bi*s_bsize:(bi+1)*s_bsize])
                loss.backward(retain_graph=True)
                lmp_optimizer.step()
                lmp_optimizer.zero_grad()
                avg_loss += loss.item()
            loss_curve.append(avg_loss / (bi+1))

            # write to file
            s = ''
        with open(write_fname, 'a+') as f:
            new_dist = np.around(new_dist.cpu().numpy(), 2).tolist()
            for l in new_dist:
                s += '\t'.join([str('%.5f' % x) for x in l]) + '\n'
            s += '\n'.join([str('%.5f' % x) for x in loss_curve])
            f.write(s)

        # set dropout back to 0.1
        self.modules['model'].train()
        self.disable_train_LMP(self.modules['model'])

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
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()   

        return self.lr_scheduler.get_lr()[0]

    def calculate_gradient(self, batch, curr_step):
        ''' Runs one step of optimization '''
        # run the data through the model
        self.model.train()

        # linear scheduling of tradeoff
        step_progress = (curr_step + 0.) / self.config.max_steps if self.config.linear_tradeoff else 0

        loss, nll, reward, sum_layermask = self.model(batch, step_progress)

        # nn.DataParallel wants to gather rather than doing a reduce_add, so the output here
        # will be a tensor of values that must be summed
        nll = nll.sum()
        loss = loss.sum()

        # calculate gradients then run an optimization step
        loss.backward(retain_graph=True)

        # need to use .item() which converts to Python scalar
        # because as a Tensor it accumulates gradients
        return nll.item(), torch.sum(batch['target_lens']).item(), 0, sum_layermask.item()

    def __call__(self, start_epoch, experiment, verbose=0):
        ''' Execute training '''
        with ExitStack() as stack:
            stack.enter_context(chunked_scattering())
            stack.enter_context(experiment.train())

            if start_epoch > 0 or experiment.curr_step > 0:
                # TODO: Hacky approach to decide if the metric store should be loaded. Revisit later
                self.metric_store = self.metric_store.load()

            epoch = start_epoch
            experiment.log_current_epoch(epoch)
            while not self.is_done(experiment, epoch):
                experiment.log_current_epoch(epoch)
                print("============epoch %i===========" % epoch)
                self.train_epoch(epoch, experiment, verbose)
                experiment.log_epoch_end(epoch)
                epoch += 1

            if self.stopped_early:
                print('Stopping early!')
            else:
                new_best = False
                if self.config.early_stopping:
                    new_best = self.evaluate(experiment, epoch, verbose)

                self.checkpoint(epoch, experiment.curr_step, new_best)
