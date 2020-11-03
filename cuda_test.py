#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cuda_test.py: This file contains function to check 
the version and availability of cuda on the system.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import torch
import sys

from subprocess import call

def cuda_test():
    """ Function used to give informations about the environment and the available GPUs """
    # This flag enable the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    print('\n__Python VERSION :', sys.version)
    print('__pyTorch VERSION :', torch.__version__)
    print('__CUDA VERSION : ', torch.version.cuda)
    print('__CUDNN VERSION : ', torch.backends.cudnn.version())
    print('__Number CUDA Devices : ', torch.cuda.device_count())
    print('__Devices : ')

    call(["nvidia-smi", "--format=csv", 
        "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])

    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    return torch.cuda.is_available()

def get_device(use_cuda):
    use_cuda = use_cuda and torch.cuda.is_available()
    print('\n' + '=' * 30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('=' * 30 + '\n')
    return torch.device("cuda" if use_cuda else "cpu")