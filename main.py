'''
SynST

--
Main entry point for training the SynST
'''

from __future__ import print_function

import sys
import copy
import threading
from contextlib import ExitStack

# comet_ml now fails to initialize if any torch module
# is loaded before it... so make sure it's loaded first
# pylint:disable=unused-import
import comet_ml
# pylint:enable=unused-import

import torch
from torch.autograd import profiler, set_detect_anomaly

from args import parse_args
from data.utils import get_dataloader
from models.utils import restore, init_indices_q, init_attended_indices, encoder_indices_matq, decoder_indices_matq, encoder_attended_indices, decoder_attended_indices
from models.utils import init_indices
from utils import profile

# import comet_ml in the top of your file
from comet_ml import Experiment
import pdb
    
# Add the following code anywhere in your machine learning file


def main(argv=None):
    ''' Main entry point '''
    args = parse_args(argv)

    # initialize indices_matq

    init_indices(args)

    print(f'Running torch {torch.version.__version__}')

    profile_cuda_memory = args.config.cuda.profile_cuda_memory
    pin_memory = 'cuda' in args.device.type and not profile_cuda_memory
    dataloader = get_dataloader(
        args.config.data, args.seed_fn, pin_memory,
        args.num_devices, shuffle=args.shuffle
    )
    print(dataloader.dataset.stats)

    args.config.model.action_type = args.action_type
    model = args.model(args.config.model, dataloader.dataset)
    action = args.action(args.action_config, model, dataloader, args.device)
    if args.action_type == 'train' and args.action_config.early_stopping:
        args.config.data.split = 'valid'
        args.config.data.max_examples = 0
        action.validation_dataloader = get_dataloader(
            args.config.data, args.seed_fn, pin_memory,
            args.num_devices, shuffle=args.shuffle
        )
    # pdb.set_trace()

    if args.action_type == "iterative_train" and not args.action_config.debug:
        args.config.data.split = 'valid'
        args.config.data.max_examples = 0
        args.config.data.batch_size = args.action_config.sample_batch_size
        args.config.data.batch_method = "example"
        action.validation_dataloader = get_dataloader(
            args.config.data, args.seed_fn, pin_memory,
            args.num_devices, shuffle=True
        )

    if args.action_type == "iterative_train" and args.action_config.debug: # if debug oracle-sample based experiment, train on valid and test on test
        args.config.data.split = 'test'
        args.config.data.max_examples = 0
        args.config.data.batch_size = args.action_config.sample_batch_size
        args.config.data.batch_method = "example"
        action.validation_dataloader = get_dataloader(
            args.config.data, args.seed_fn, pin_memory,
            args.num_devices, shuffle=True
        )


    if args.config.cuda.profile_cuda_memory:
        print('Profiling CUDA memory')
        memory_profiler = profile.CUDAMemoryProfiler(
            action.modules.values(),
            filename=profile_cuda_memory
        )

        sys.settrace(memory_profiler)
        threading.settrace(memory_profiler)

    step = 0
    epoch = 0
    if args.restore:
        restore_modules = {
            module_name: module
            for module_name, module in action.modules.items()
            if module_name not in args.reset_parameters
        }

        epoch, step = restore(
            args.restore,
            restore_modules,
            num_checkpoints=args.average_checkpoints,
            map_location=args.device.type,
            strict=not args.reset_parameters
        )

        model.reset_named_parameters(args.reset_parameters)
        if 'step' in args.reset_parameters:
            step = 0
            epoch = 0

    args.experiment.set_step(step)

    with ExitStack() as stack:
        stack.enter_context(profiler.emit_nvtx(args.config.cuda.profile_cuda))
        stack.enter_context(set_detect_anomaly(args.detect_anomalies))
        action(epoch, args.experiment, args.verbose)


if __name__ == '__main__':
    main()

