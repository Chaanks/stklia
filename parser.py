#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parser.py: This file contains function used to
parse the command and the config file.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import time
import json
import argparse
import configparser

import numpy as np 


from pprint import pprint
from pathlib import Path

def check_file_exist(file):
    assert file.exists(), f"No sich file {file.name}"
    return file

def fetch_args_and_config(verbose=0):
    parser = argparse.ArgumentParser(description='Train and test of ResNet for speaker verification')

    parser.add_argument('--cfg', type=str, help="Path to a config file")
    parser.add_argument('--checkpoint', '--resume-checkpoint', type=int, default=-1, # which model to use, overidden by 'best'
                            help='Use model checkpoint, default -1 uses final model')

    args = parser.parse_args()
    # Check that there is a config file
    assert args.cfg, "Please specify a config file using --cfg"
    args.cfg = Path(args.cfg)
    assert args.cfg.is_file(), f"No such file {args.cfg}"    

    args._start_time = time.ctime()

    # Parse the config file :
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.generator_lr        = config['Hyperparams'].getfloat('generator_lr', fallback=0.2)
    args.classifier_lr       = config['Hyperparams'].getfloat('classifier_lr', fallback=0.2)
    args.batch_size          = config['Hyperparams'].getint('batch_size', fallback=400)
    args.max_seq_len         = config['Hyperparams'].getint('seq_len', fallback=400)
    args.no_cuda             = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed                = config['Hyperparams'].getint('seed', fallback=123)
    args.num_iterations      = config['Hyperparams'].getint('num_iterations', fallback=50000)
    args.momentum            = config['Hyperparams'].getfloat('momentum', fallback=0.9)
    args.scheduler_steps     = np.array(json.loads(config.get('Hyperparams', 'scheduler_steps'))).astype(int)
    args.scheduler_lambda    = config['Hyperparams'].getfloat('scheduler_lambda', fallback=0.5)
    args.multi_gpu           = config['Hyperparams'].getboolean('multi_gpu', fallback=False)
    args.classifier_lr_mult  = config['Hyperparams'].getfloat('classifier_lr_mult', fallback=1.)

    try:
        args.model_dir           = Path(config['Inputs']['model_dir'])
        args.trained_gen_path    = check_file_exist(Path(config['Inputs']['trained_gen']))
        args.trained_class_path  = check_file_exist(Path(config['Inputs']['trained_class']))
    except KeyError:
        pass

    args.output_dir          = Path(config['Outputs']['output_dir'])
    args.checkpoints_dir     = args.output_dir / 'checkpoints/'
    args.log_file            = args.output_dir / 'train.log'
    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval')
    args.log_interval        = config['Outputs'].getfloat('log_interval', fallback=100)

    if verbose:
        pprint(vars(args))

    return args