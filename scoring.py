#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import argparse
import pprint
import random

from loguru import logger
from pathlib import Path
from simple_slurm import Slurm

EMOJI = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐸', '🐵', '🐔', '🐧', '🐦', '🐤', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🐛', '🦋', '🐌', '🐞', '🐜', '🦟', '🦗', '🕷', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀', '🐠']

if __name__ == "__main__":

    # ARGUMENTS PARSING
    parser = argparse.ArgumentParser(description='Extraction and scoring for speaker verification model')

    parser.add_argument('--cfg', type=str, required=True, help="Path to the config file")
    parser.add_argument('--checkpoint', '--resume-checkpoint', type=int, default=-1,
                            help="Model Checkpoint to use. [TEST] default : use the last one [TRAIN] default : None used, -1 : use the last one")
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    args = parser.parse_args() 

    # Check that the config file exist
    args.cfg = Path(args.cfg)
    assert args.cfg.is_file(), f"No such file {args.cfg}"

    # Load the config file
    with open(args.cfg) as file:  
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    if args.verbose:
        print(f'Config file {args.cfg} : ')
        pprint.pprint(cfg)

    for k, v in cfg['data'].items():
        # SLURM 
        slurm = Slurm()
        slurm.add_arguments(ntasks='1')
        #slurm.add_arguments(array=range(0, 2))
        slurm.add_arguments(cpus_per_task='8')
        slurm.add_arguments(partition='gpu')
        if 'sitw' in k:
            slurm.add_arguments(gpus_per_node='tesla_p100:1')
        else:
            slurm.add_arguments(gpus_per_node='1')
        #slurm.add_arguments(gpus_per_node='1')
        slurm.add_arguments(job_name=random.choice(EMOJI))
        slurm.add_arguments(output=r'slurm/logs/%j.out')
        slurm.add_arguments(mem='16G')
        slurm.add_arguments(time='192:00:00')

        slurm.sbatch(f"python extract.py -m {cfg['model_dir']} -o {cfg['out_dir']} -d {v} --checkpoint {args.checkpoint}")