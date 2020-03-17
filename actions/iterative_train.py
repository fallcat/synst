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
from torch.utils.data import DataLoader
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
import pickle
import jsonlines
import json


class IterativeTrainer(object):
    ''' An object that encapsulates model training '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.stopped_early = False
        self.dataloader = dataloader
        self.validation_dataloader = dataloader
        self.test_dataloader = None
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
            self.optimizer = optim.SGD(params, lr=config.base_lr, momentum=0.9)
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                DummyLRSchedule(
                    config.base_lr
                )
            )

        elif self.config.optimizer == "adam-fixed":
            print("using optimizer: adam with fixed learning rate")
            self.optimizer = optim.Adam(params, config.base_lr, betas=(0.9, 0.98), eps=1e-9)
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
        self.metric_store.add(metrics.Metric('train_loss', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('val_loss', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('percent_ge', metrics.format_float, max_history=1))
        self.metric_store.add(metrics.Metric('test_bleu', metrics.format_float, max_history=1))
        self.metric_store.add(metrics.Metric('num_layer', metrics.format_float, max_history=1))

        self.modules = {
            'model': model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }

        self.lmp_optimizer = optim.Adam(self.modules['model'].layer_mask_predictor.parameters(), lr=self.config.base_lr, betas=(0.9, 0.98), eps=1e-9)
        self.lmp_lr_scheduler = LambdaLR(self.lmp_optimizer, LinearLRSchedule(
                                self.config.base_lr,
                                1e-5,
                                self.config.max_train_lmp_epochs * 1999 / config.batch_size))


    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    def train_epoch(self, epoch, experiment, verbose=0):

        batches = tqdm(
            self.dataloader,
            unit='batch',
            desc="",
            file=sys.stdout 
        )

        with tqdm_wrap_stdout():
            if not self.config.test:
                self.iter_train(epoch, experiment)
            else:
                self.run_test()
            return

    def get_translated(self, translator, batches, layermask=None):

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

    def run_test(self):
        model = self.modules['model']
        model.eval()
        self.disable_train_LMP(model)
        num_layer = 2 * len(model.encoders)
        test_batches = [b for b in iter(self.test_dataloader)]

        model.set_LMP_type('noskip')
        noskip_translator = model.translator(self.config).to(torch.device("cuda"))
        test_gold, test_allon_gen, _ = self.get_translated(noskip_translator, test_batches)
        test_allon_bleu, test_allon_bleu_by_sent = self.get_bleu_res(test_allon_gen, test_gold)

        model.set_LMP_type('itertrain')
        sample_translator = model.translator(self.config).to(torch.device("cuda"))
        test_gen_gold, test_gen, test_masks = self.get_translated(sample_translator, test_batches)
        test_gen_bleu, test_gen_bleu_by_sent = self.get_bleu_res(test_gen, test_gold)
        
        percent_g = sum([int(a >= b) for a, b in zip(test_gen_bleu_by_sent, test_allon_bleu_by_sent)]) / len(test_allon_gen)
        combined_test_gen = [test_gen[i] if test_gen_bleu_by_sent[i] > test_allon_bleu_by_sent[i] else test_allon_gen[i] for i in range(len(test_gen))]
        combined_test_bleu = sacrebleu.corpus_bleu(combined_test_gen, [test_gold], tokenize='none').score
        ratio = sum([a.sum(dim=0) for a in test_masks])/(len(test_masks) * test_masks[0].shape[0])
        all_on_masks = sum([(m.sum(dim=1) == num_layer).sum() for m in test_masks])
        total_masks = sum([m.shape[0] for m in test_masks])
        
        filter_allon = [m.sum(dim=1) != num_layer for m in test_masks] 
        non_allon_configs = [m.float()[:, None]*lm for m, lm in zip(filter_allon, test_masks)]
        non_allon_config_layers = sum([m.sum() for m in non_allon_configs])
        
        filter_allon = torch.cat(filter_allon)
        non_allon_gen = [test_gen[i] for i in range(len(test_gen)) if filter_allon[i]]
        non_allon_allon = [test_allon_gen[i] for i in range(len(test_allon_gen)) if filter_allon[i]]
        non_allon_gold = [test_gold[i] for i in range(len(test_gold)) if filter_allon[i] ]
        non_allon_gen_bleu = sacrebleu.corpus_bleu(non_allon_gen, [non_allon_gold], tokenize='none').score
        non_allon_allon_bleu = sacrebleu.corpus_bleu(non_allon_allon, [non_allon_gold], tokenize='none').score

        total_selected_layers = sum([m.sum().item() for m in test_masks])
        
        print("test corpus bleu: {:.2f}".format(test_gen_bleu))
        print("percent >= : {}".format(percent_g))
        print("combined test corpus bleu: {:.2f}".format(combined_test_bleu))
        print("layer selection ratio: {}".format(np.around(ratio.cpu().numpy(), 2).tolist()))
        print("all-on ratio: {}".format(all_on_masks.item() / float(total_masks))) # what percent of the test set selecting all-on config
        print("average #layer non-all-on config example {}".format(non_allon_config_layers / (total_masks - all_on_masks.item() + 1e-10)))
        print("average #layer all {}".format(total_selected_layers / float(total_masks)))
        print("non-allon config: bleu using layers selected by LMP {}".format(non_allon_gen_bleu))
        print("non-allon config: bleu using layers selected by LMP {}".format(non_allon_allon_bleu))

        fname = os.path.join(self.config.checkpoint_directory, 'translated_lmp.txt')
        with open(fname, 'w') as f:
            for l in test_gen:
                f.write(l + '\n')

        fname = os.path.join(self.config.checkpoint_directory, 'test_masks.pkl')
        with open(fname, 'wb') as f:
            pickle.dump({'test_masks': test_masks, 
                         'test_lmp_bleu_by_sent': test_gen_bleu_by_sent, 
                         'test_allon_bleu_by_sent': test_allon_bleu_by_sent}, f)

    def get_bleu_res(self, decoded, gold, tokenize='none'):

        corpus_bleu = sacrebleu.corpus_bleu(decoded, [gold], tokenize=tokenize).score
        bleu_by_sent = [sacrebleu.corpus_bleu([gen_i], [[gold_i]], tokenize='none').score for gen_i, gold_i in zip(decoded, gold)]

        return corpus_bleu, bleu_by_sent

    def get_loss(self, model, batch):

        embedding = model.embed(batch['inputs'].cuda(), model.embedding).detach()
        padding_masks = batch['inputs'].eq(model.padding_idx).cuda()
        loss = model.layer_mask_predictor(embedding, padding_masks, aggregate_stats=batch['y1'].cuda())

        return loss

    def optimize(self, loss):

        loss.backward()
        self.lmp_optimizer.step()
        self.lmp_optimizer.zero_grad()
        self.lmp_lr_scheduler.step()

    def validate(self, model, valid_batches):
        val_loss = 0
        for vi, v_batch in enumerate(valid_batches):
            embedding = model.embed(v_batch['inputs'].cuda(), model.embedding).detach()
            padding_masks = v_batch['inputs'].eq(model.padding_idx).cuda()
            loss = model.layer_mask_predictor(embedding, padding_masks, aggregate_stats=v_batch['y1'].cuda())
            val_loss += loss
        val_loss = val_loss.item()/(vi+1)
        return val_loss

    def iter_train(self, epoch, experiment, verbose=1):
        """
        Input:
            experiment

        data:
            dataloader: valid-train
            validation_dataloader: valid-val
            test_dataloader: test
        """

        model = self.modules['model']
        model.eval()
        num_layer = 2 * len(model.encoders)
        
        model.set_LMP_type('noskip')
        noskip_translator = model.translator(self.config).to(torch.device("cuda"))
        valid_batches = [b for b in iter(self.validation_dataloader)] # bc valid batches is a lot smaller than test batches
        _, val_allon, _ = self.get_translated(noskip_translator, valid_batches)

        model.set_LMP_type('itertrain')
        sample_translator = model.translator(self.config).to(torch.device("cuda"))
        self.enable_train_LMP(model)
        train_losses = self.metric_store['train_loss']
        val_losses = self.metric_store['val_loss']
        percent_ges = self.metric_store['percent_ge']
        test_bleus = self.metric_store['test_bleu']
        average_num_layers = self.metric_store['num_layer']

        for epoch_i in range(epoch, epoch+self.config.max_train_lmp_epochs):

            end_flag = False
            for bi, batch in enumerate(iter(self.dataloader)):
                experiment.set_step(experiment.curr_step + 1)

                train_loss = self.get_loss(model, batch)
                self.optimize(train_loss)
                train_losses.update(train_loss.item())
                experiment.log_metric('train_loss', train_losses.last_value)

                if experiment.curr_step % self.config.eval_every == 0:
                    with torch.no_grad():
                        val_loss = self.validate(model, valid_batches)
                        val_losses.update(val_loss)
                        experiment.log_metric('val_loss', val_losses.last_value)

                        print("step: %i epoch : %i batch : %i val loss: %f" % (experiment.curr_step, epoch_i, bi, val_loss))
                        if self.end_itertrain(val_losses.values, self.config.early_stopping) and epoch_i > epoch + 1:
                            self.checkpoint(epoch, experiment.curr_step, False)
                            end_flag = True
                            break

                        if self.is_best_checkpoint(val_losses) and epoch_i > epoch + 4:
                            self.checkpoint(epoch, experiment.curr_step, True)

            if end_flag:
                break

    def is_best_checkpoint(self, val_losses):
        """best checkpoint or not"""
        if val_losses.values[-1] == min(val_losses.values):
            return True
        else:
            return False

    def end_itertrain(self, val_losses, early_stopping):
        """decide when to stop iterative training"""
        if all([val_losses[-1] > v for v in val_losses[-early_stopping:-1]]):
            return True
        else:
            return False


    def enable_train_LMP(self, model):
        model.layer_mask_predictor.eval = False
        for pname, pval in model.named_parameters():
            if 'layer_mask_predictor' in pname:
                pval.requires_grad = True
                continue
        

    def disable_train_LMP(self, model):
        model.layer_mask_predictor.eval = True
        for pname, pval in model.named_parameters():
            if 'layer_mask_predictor' in pname:
                pval.requires_grad = True
                continue
        
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


    def __call__(self, start_epoch, experiment, verbose=0):
        ''' Execute training '''
        with ExitStack() as stack:
            stack.enter_context(chunked_scattering())
            stack.enter_context(experiment.train())

            epoch = start_epoch
            experiment.log_current_epoch(epoch)
            self.train_epoch(epoch, experiment, verbose)


