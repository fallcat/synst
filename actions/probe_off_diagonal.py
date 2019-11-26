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
from torch import nn
from tqdm import tqdm

from utils import profile
from utils import tqdm_wrap_stdout
from models.utils import save_attention


class ProbeOffDiagonal(object):
    ''' An object that probes sentences that focus off diagonal '''
    CURRENT = None

    def __init__(self, config, model, dataloader, device):
        self.config = config
        self.dataloader = dataloader
        self.translator = model.translator(config).to(device)
        self.model = model
        self.device = device

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.modules = {
            'model': model
        }

        self.off_diagonal = []
        self.non_off_diagonal = []

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

    def translate_all(self, output_file, epoch, experiment, verbose=0):
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
            self.model.eval()
            ordered_outputs = []
            for batch in batches:
                # print("in probe new translate", flush=True)
                # run the data through the model
                batches.set_description_str(get_description())
                sequences = self.translator.translate(batch)

                if self.config.timed:
                    continue

                target_sequences = next(iter(sequences.values()))
                new_targets = []
                output_sentences = []
                source_sentences = []
                for i, example_id in enumerate(batch['example_ids']):
                    outputs = []
                    if verbose > 1:
                        trim = verbose < 2
                        join = verbose < 3
                        for key in sequences.keys():
                            sequence = sequences[key][i]
                            sequence = ' '.join(self.dataset.decode(sequence, join, trim))
                            outputs.append(f'{key}: {sequence}\n')
                        outputs.append(f'+++++++++++++++++++++++++++++\n')
                    else:
                        sequence = target_sequences[i]
                        new_targets.append(torch.LongTensor(sequence))
                        decoded = ' '.join(self.dataset.decode(sequence, trim=not verbose))
                        outputs.append(f'{decoded}\n')
                        output_sentence = ' '.join(self.dataset.decode(sequence, join=False, trim=not verbose))
                        output_sentences.append(output_sentence)
                        source_sentence = ' '.join(self.dataset.decode(batch['inputs'][i], join=False, trim=not verbose))
                        source_sentences.append(source_sentence)

                        # Encoder heatmap
                        # print("saving encoder heatmap")
                        # for j in range(encoder_attn_weights_tensor.shape[0]):
                        #     for k in range(encoder_attn_weights_tensor.shape[1]):
                        #         attn_filename = f'encoder_attn_weights{example_id}_l{j}_h{k}.png'
                        #         attn_path = os.path.join(self.config.output_directory, attn_filename)
                        #         save_attention(source_sentence, source_sentence,
                        #                        encoder_attn_weights_tensor[j][k].cpu().numpy(), attn_path)

                    if self.config.order_output:
                        ordered_outputs.append((example_id, outputs))
                    else:
                        output_file.writelines(outputs)

                self.dataset.collate_field(batch, 'target', new_targets)
                result = self.model(batch)
                # Decoder heatmap
                # print("saving decoder heatmap")
                for i, example_id in enumerate(batch['example_ids']):
                    print("result['enc_dec_attn_weights_tensor']", result['enc_dec_attn_weights_tensor'].shape)
                    attn_weights_shape = result['enc_dec_attn_weights_tensor'].shape
                    attn_weights = result['enc_dec_attn_weights_tensor'].view(-1,
                                                                              attn_weights_shape[2],
                                                                              attn_weights_shape[3])
                    indices_q = torch.arange(attn_weights_shape[2], dtype=torch.float32,
                                             device=attn_weights.get_device()).view(1, -1) * self.dataset.word_count_ratio
                    argmax_weights = torch.argmax(attn_weights, dim=2).type_as(indices_q)
                    max_weights = attn_weights[argmax_weights]
                    print("max_weights", max_weights.shape)
                    distance = torch.abs(argmax_weights - indices_q)
                    print("attn_weights", attn_weights.shape)
                    print("distance", distance.shape)
                    print("distance >= 1", (distance >= 1).shape)
                    print("attn_weights[distance >= 1]", argmax_weights[distance >= 1].shape)
                    max_prob = torch.max(argmax_weights[distance >= 1])
                    argmax_offset = torch.max(distance)
                    number = torch.sum(distance >= 1)
                    #self.config.off_diagonal_threshold_param

                    if self.config.off_diagonal_threshold_type == "number":
                        print("number")
                        if 1 <= self.config.off_diagonal_threshold_param <= number \
                                or self.config.off_diagonal_threshold_param < 1 \
                                and number / float(attn_weights_shape[2]) >= self.config.off_diagonal_threshold_param:
                            self.off_diagonal.append(example_id)
                            print("in")
                        else:
                            self.non_off_diagonal.append(example_id)
                            print("out")
                    elif self.config.off_diagonal_threshold_type == "offset":
                        print("offset")
                        if argmax_offset >= self.config.off_diagonal_threshold_param:
                            self.off_diagonal.append(example_id)
                        else:
                            self.non_off_diagonal.append(example_id)
                    else:  # prob
                        print("prob")
                        if max_prob >= self.config.off_diagonal_threshold_param:
                            self.off_diagonal.append(example_id)
                        else:
                            self.non_off_diagonal.append(example_id)

                    #self.dataset.word_count_ratio

                    # for j in range(result['decoder_attn_weights_tensor'].shape[0]):
                    #     for k in range(result['decoder_attn_weights_tensor'].shape[1]):
                    #         attn_filename = f'decoder_attn_weights{example_id}_l{j}_h{k}.png'
                    #         attn_path = os.path.join(self.config.output_directory, attn_filename)
                    #         save_attention(output_sentences[i], output_sentences[i],
                    #                        result['decoder_attn_weights_tensor'][j][k].cpu().numpy(), attn_path)
                    #         attn_filename = f'enc_dec_attn_weights{example_id}_l{j}_h{k}.png'
                    #         attn_path = os.path.join(self.config.output_directory, attn_filename)
                    #         save_attention(source_sentences[i], '<PAD>' + output_sentences[i],
                    #                        result['enc_dec_attn_weights_tensor'][j][k].cpu().numpy(), attn_path)

            print("num off diagonal", len(self.off_diagonal))
            print("num non off diagonal", len(self.non_off_diagonal))

            for _, outputs in sorted(ordered_outputs, key=lambda x: x[0]): # pylint:disable=consider-using-enumerate
                output_file.writelines(outputs)

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
                ProbeOffDiagonal.CURRENT = self
                stmt = f'Translator.CURRENT.translate_all(None, {epoch}, None, {verbose})'
                timing = timeit.timeit(stmt, stmt, number=self.config.timed, globals=globals())
                print(f'Translation timing={timing/self.config.timed}')
            else:
                step = experiment.curr_step
                output_filename = self.config.output_filename or f'translated_{step}.txt'
                output_path = os.path.join(self.config.output_directory, output_filename)
                output_file = stack.enter_context(open(output_path, 'wt'))

                if verbose:
                    print(f'Outputting to {output_path}')

                self.translate_all(output_file, epoch, experiment, verbose)
