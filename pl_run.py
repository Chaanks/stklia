import os
import time
import shutil
import numpy as np
import argparse

from loguru import logger
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

import dataset
from parser import fetch_config
from cuda_test import cuda_test, get_device
from train_resnet import train
from test_resnet import score_utt_utt
from pl_model import NNET

if __name__ == "__main__":

    os.environ["SLURM_NODELIST"] = os.environ["SLURM_NODELIST"].replace(",", " ")

    # ARGUMENTS PARSING
    parser = argparse.ArgumentParser(description='Train and test of ResNet for speaker verification')
    parser.add_argument('--cfg', type=str, required=True, help="Path to a config file")
    
    args = parser.parse_args()

    # CONFIG FILE PARSING
    args = fetch_config(args, 1)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints_dir.mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    cuda_test()
    device = get_device(not args.no_cuda)

    if args.log_file.exists():
        args.log_file.unlink()
    logger.add(args.log_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", backtrace=False, diagnose=False)

    # TODO
    args.num_classes = 5994
    print(args)
    model = NNET(args)

    #trainer = Trainer(gpus=4, accelerator='ddp', plugins='ddp_sharded', max_epochs=2, check_val_every_n_epoch=1, num_sanity_val_steps=0)
    #multi nodes OP CHEAT AS FUCK distributed_backend='ddp'
    #trainer = Trainer(gpus=2, accelerator='ddp', plugins='ddp_sharded', num_nodes=2, max_epochs=2, check_val_every_n_epoch=1, num_sanity_val_steps=0)
    trainer = Trainer(gpus=2, accelerator='ddp_sharded', plugins='ddp_sharded', num_nodes=2, max_epochs=2)
    trainer.fit(model)