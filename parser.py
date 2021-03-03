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
import configparser

import numpy as np 
from pprint import pprint
from pathlib import Path

def check_file_exist(file):
    assert file.exists(), f"No sich file {file.name}"
    return file

def fetch_config(args, verbose=False):
    
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

    args.features_per_frame  = config['Dataset'].getint('features_per_frame', fallback=30)
    # try to parse a train dataset
    try:
        args.train_data_path = [Path(p) for p in config['Dataset']['train'].split()]
    except KeyError:
        args.train_data_path = None
    # try to parse a eval dataset
    try:
        args.eval_data_path  = [Path(p) for p in config['Dataset']['eval'].split()]
        args.eval_trials_path = [Path(p) for p in config['Dataset']['eval_trials'].split()]
    except KeyError:
        args.eval_data_path  = None
        args.eval_trials_path = None
    # try to parse a test dataset
    try:
        args.test_data_path  = [Path(p) for p in config['Dataset']['test'].split()]
        args.test_trials_path = [Path(p) for p in config['Dataset']['test_trials'].split()]
    except KeyError:
        args.test_data_path  = None
        args.test_trials_path = None

    args.emb_size = config['Model'].getint('emb_size', fallback=256)
    args.layers = np.array(json.loads(config.get('Model', 'layers'))).astype(int)
    args.num_filters = np.array(json.loads(config.get('Model', 'num_filters'))).astype(int)
    args.zero_init_residual = config['Model'].getboolean('zero_init_residual', fallback=False)
    args.pooling = config['Model'].get('pooling', fallback='statistical')
    assert args.pooling in ['min', 'max', 'mean', 'std', 'statistical', 'std_skew', 'std_kurtosis'], f"Unknow pooling mode {args.pooling}"

    args.model_dir           = Path(config['Outputs']['model_dir'])
    args.checkpoints_dir     = args.model_dir / 'checkpoints/'
    args.log_file            = args.model_dir / 'train.log'

    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval')
    args.log_interval        = config['Outputs'].getfloat('log_interval', fallback=100)

    if verbose:
        pprint(vars(args))

    return args
