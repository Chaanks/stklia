#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset.py: This file contains funtions to read kaldi files.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import random

import pandas as pd
import numpy as np
import torch

# I/O fonctions :
def load_n_col(file):
    """ 
        Used to load files like utt2spk
        Using pandas for better perf (3x faster)
    """
    df = pd.read_csv(file, delimiter=" ", header=None)
    columns = [list(df[col]) for col in df]
    return columns

def load_one_tomany(file):
    one2many = {}
    with open(file) as fp:
        for line in fp:
            _ = line.strip().split(' ', 1)
            one2many[_[0]] = _[1].split(' ')
    return one2many

def read_scp(file):
    df = pd.read_csv(file, delimiter=" ", header=None)
    return {k:v for k, v in zip(df[0], df[1])}

def read_utt_list(filename):
	df = pd.read_csv(filename, delimiter=" ", header=None)
	return df[0].tolist()

def get_spk_from_utt(utt):
	""" Edit this function to fit your dataset """
	return utt.split('-')[0]

def utt_list_to_utt2spk(utt_list):
	utt2spk = {}
	for utt in utt_list:
		utt2spk[utt] = get_spk_from_utt(utt)
	return utt2spk

def utt_list_to_spk2utt(utt_list):
	spk2utt = {}
	for utt in utt_list:
		spk = get_spk_from_utt(utt)
		try:
			spk2utt[spk].append(utt)
		except KeyError:
			spk2utt[spk] = [utt]
	return spk2utt

def train_transform(feats, seqlen):
	leeway = feats.shape[0] - seqlen
	startslice = np.random.randint(0, int(leeway)) if leeway > 0 else 0
	if leeway > 0:
		feats = feats[startslice:startslice + seqlen] 
	else:
		feats = np.pad(feats, [(0, -leeway), (0, 0)], 'constant')
	return torch.FloatTensor(feats)
    
def test_transform(feats, seqlen):
	leeway = feats.shape[0] - seqlen
	startslice = 0
	feats = feats[startslice:startslice + seqlen] if leeway > 0 else feats
	return torch.FloatTensor(feats)