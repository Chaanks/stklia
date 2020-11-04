#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py: This file is used as a launcher to train or test the resnet.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"


import time
import shutil
import numpy as np
import argparse

from loguru import logger
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import dataset
from parser import fetch_config
from cuda_test import cuda_test, get_device
from train_resnet import train
from test_resnet import score_utt_utt
from models import resnet34

if __name__ == "__main__":

    # ARGUMENTS PARSING
    parser = argparse.ArgumentParser(description='Train and test of ResNet for speaker verification')

    parser.add_argument("-m", "--mode", type=str, choices=["train", "test"], required=True, help="Put this argument to train the resnet")
    parser.add_argument('--cfg', type=str, required=True, help="Path to a config file")
    parser.add_argument('--checkpoint', '--resume-checkpoint', type=int, default=-2,
                            help="Model Checkpoint to use. [TEST] default : use the last one [TRAIN] default : None used, -1 : use the last one")

    args = parser.parse_args()

    # Check that the config file exist
    args.cfg = Path(args.cfg)
    assert args.cfg.is_file(), f"No such file {args.cfg}"

    # CONFIG FILE PARSING
    args = fetch_config(args, 1)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints_dir.mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    cuda_test()
    device = get_device(not args.no_cuda)

    # TRAIN
    if args.mode == "train":
        assert args.train_data_path, "No training dataset given in train mode"
        ds_train = dataset.make_kaldi_ds(args.train_data_path, seq_len=args.max_seq_len, evaluation=False, trials=None)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=8)

        if args.eval_data_path and args.eval_trial_path:
            ds_val = dataset.make_kaldi_ds(args.eval_data_path, seq_len=None, evaluation=True, trials=args.eval_trial_path)
        else:
            ds_val = None
            
        if args.resume_checkpoint < 0:
            shutil.copy(args.cfg, args.model_dir / 'experiment_settings.cfg')
        else:
            shutil.copy(args.cfg, args.model_dir / 'experiment_settings_resume.cfg')
            
        if args.log_file.exists():
            args.log_file.unlink()
        logger.add(args.log_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", backtrace=False, diagnose=False)

        train(args, dl_train, device, ds_val)

    # TEST
    if args.mode == "test":
        assert args.test_data_path and args.test_trial_path, "No test dataset or trials given in test mode"
        ds_test = dataset.make_kaldi_ds(args.test_data_path, seq_len=None, evaluation=True, trials=args.test_trial_path)

        # Load generator
        if args.checkpoint < 0:
            g_path = args.model_dir / "final_g_{}.pt".format(args.num_iterations)
            g_path_test = g_path
        else:
            print('use checkpoint {}'.format(args.checkpoint))
            g_path = args.checkpoints_dir / "g_{}.pt".format(args.checkpoint)
            g_path_test = g_path

        model = resnet34(args)
        model.load_state_dict(torch.load(g_path), strict=False)
        model = model.to(device)

        # TODO : support score_spk_utt
        score_utt_utt(model, ds_test, device)
