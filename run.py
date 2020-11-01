#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py: This file is used as a launcher to train or test the resnet.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import time
import numpy as np

from loguru import logger

import torch
from torch.utils.data import DataLoader

import dataset
from parser import fetch_args_and_config
from cuda_test import cuda_test, get_device
from train_resnet import train
from test_resnet import score_utt_utt
from models import resnet34

if __name__ == "__main__":
    args = fetch_args_and_config(1)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints_dir.mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    cuda_test()
    device = get_device(not args.no_cuda)

    #TODO: support pytorch dataset

    if args.test_data_path and args.trials_path:
        ds_val = dataset.make_kaldi_ds(args.test_data_path, seq_len=None, evaluation=True, trials=args.trials_path)
    else:
        if args.mode == "test":
            raise KeyError("No trial or Test data in test mode")
        ds_val = None

    # TRAIN
    if args.mode == "train":
        ds_train = dataset.make_kaldi_ds(args.train_data_path, seq_len=args.max_seq_len, evaluation=False, trials=None)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)

        if args.log_file.exists():
            args.log_file.unlink()
        logger.add(args.log_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", backtrace=False, diagnose=False)

        train(args, dl_train, device, ds_val)

    #TEST
    if args.mode == "test":

        # Load generator
        if args.checkpoint < 0:
            g_path = args.model_dir / "final_g_{}.pt".format(args.num_iterations)
            g_path_test = g_path
        else:
            print('use checkpoint {}'.format(args.checkpoint))
            g_path = args.checkpoints_dir / "g_{}.pt".format(args.checkpoint)
            g_path_test = g_path

        # TODO: choose model type from cfg
        model = resnet34(args)
        model.load_state_dict(torch.load(g_path), strict=False)
        model = model.to(device)

        # TODO : support score_spk_utt
        score_utt_utt(model, ds_val, device)
