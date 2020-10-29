#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset.py: This file contains funtions to read kaldi files
and the class of our speaker dataset.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import random
import numpy as np
import pandas

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from collections import OrderedDict
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from kaldi_io import read_mat

# I/O fonctions :
def load_n_col(file):
    """ 
        Used to load files like utt2spk
        Using pandas for better perf (3x faster)
    """
    df = pandas.read_csv(file, delimiter=" ", header=None)
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
    df = pandas.read_csv(file, delimiter=" ", header=None)
    return {k:v for k, v in zip(df[0], df[1])}

def read_utt_list(filename):
	df = pandas.read_csv(filename, delimiter=" ", header=None)
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

# Dataset class
class SpeakerDataset(Dataset):
    def __init__(self, utt2path, utt2spk, spk2utt, loading_method, seq_len=None, evaluation=False, trials=None):

        self.utt2path = utt2path
        self.loading_method = loading_method
        self.utt_list = list(utt2spk.keys())

        self.utts, self.uspkrs = list(utt2spk.keys()), list(utt2spk.values())

        self.label_enc = LabelEncoder()

        self.spkrs, self.spkutts = list(spk2utt.keys()), list(spk2utt.values())
        self.spkrs = self.label_enc.fit_transform(self.spkrs)
        self.spk2utt = OrderedDict({k: v for k, v in zip(self.spkrs, self.spkutts)})

        self.uspkrs = self.label_enc.transform(self.uspkrs)
        self.utt2spk = OrderedDict({k: v for k, v in zip(self.utts, self.uspkrs)})

        self.seq_len = seq_len
        self.evaluation = evaluation
        self.num_classes = len(self.label_enc.classes_)

        self.trans = test_transform if self.evaluation else train_transform

        self.trials = trials
        # assert (self.trials == None) and (evaluation == True), "No trials given while on eval mode"

    def __repr__(self):
        return f"SpeakerDataset w/ {len(self.spk2utt)} speakers and {len(self.utt2spk)} sessions. eval={self.evaluation}"

    def __len__(self):
        if self.evaluation:
            return len(self.utt_list)
        return len(self.spk2utt)

    def __getitem__(self, idx):
        """ Returns one random utt of selected speaker """

        if self.evaluation:
            utt = self.utt_list[idx]
        else:
            utt = np.random.choice(self.spk2utt[idx])

        spk = self.utt2spk[utt]
        feats = self.loading_method(self.utt2path[utt])

        if self.seq_len:
            feats = self.trans(feats, self.seq_len)

        return feats, spk, utt

# Recettes :
def make_pytorch_ds(utt_list, utt2path_func, seq_len=400, evaluation=False, trials=None):
    """ 
    Make a SpeakerDataset from only the path of the kaldi dataset.
    This function will use the files 'feats.scp', 'utt2spk' 'spk2utt'
    present in ds_path to create the SpeakerDataset.
    """
    ds = SpeakerDataset(
        utt2path = {k:utt2path_func(k) for k in utt_list},
        utt2spk  = utt_list_to_utt2spk(utt_list),
        spk2utt  = utt_list_to_spk2utt(utt_list),
        loading_method = lambda path: torch.load(path),
        seq_len  = seq_len,
        evaluation = evaluation,
        trials=trials,
    )
    return ds

#TODO: remove make_kaldi_ds_from_mul_path and add the feature in this make_kaldi_ds function
def make_kaldi_ds(ds_path, seq_len=400, evaluation=False, trials=None):
    """ 
    Make a SpeakerDataset from only the path of the kaldi dataset.
    This function will use the files 'feats.scp', 'utt2spk' 'spk2utt'
    present in ds_path to create the SpeakerDataset.
    """
    if not isinstance(ds_path, list):
        ds_path = [ds_path]
    
    utt2spk, spk2utt, utt2path = {}, {}, {}
    for i, path in enumerate(ds_path):
        utt2path.update(read_scp(path / 'feats.scp'))
        utt2spk.update(read_scp(path / 'utt2spk'))
        # can't do spk2utt.update(t_spk2utt) as update is not additive
        t_spk2utt = load_one_tomany(path / 'spk2utt')
        for spk, utts in t_spk2utt.items():
            try:
                spk2utt[spk] += utts
            except KeyError:
                spk2utt[spk] = utts

    ds = SpeakerDataset(
        utt2path = utt2path,
        utt2spk  = utt2spk,
        spk2utt  = spk2utt,
        loading_method = lambda path: torch.FloatTensor(read_mat(path)),
        seq_len  = seq_len,
        evaluation = evaluation,
        trials=trials,
    )
    return ds

if __name__ == "__main__":
    from pathlib import Path
    print(make_kaldi_ds(Path("/local_disk/arges/jduret/kaldi/egs/fabiol/v2/data/fabiol_test_no_sil")))
    print(make_kaldi_ds([Path("/local_disk/arges/jduret/kaldi/egs/fabiol/v2/data/fabiol_test_no_sil"),
                    Path("/local_disk/arges/jduret/kaldi/egs/fabiol/v2/data/fabiol_enroll_no_sil")]))