#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset.py: This file contains funtions and class of our speaker dataset.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

from pathlib import Path
from collections import OrderedDict

import data_io

import torch
import numpy as np
from kaldi_io import read_mat
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


# Dataset class
class SpeakerDataset(Dataset):
    """ Characterizes a dataset for Pytorch """
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

        self.trans = data_io.test_transform if self.evaluation else data_io.train_transform

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
        utt2spk  = data_io.utt_list_to_utt2spk(utt_list),
        spk2utt  = data_io.utt_list_to_spk2utt(utt_list),
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
    for _ , path in enumerate(ds_path):
        utt2path.update(data_io.read_scp(path / 'feats.scp'))
        utt2spk.update(data_io.read_scp(path / 'utt2spk'))
        # can't do spk2utt.update(t_spk2utt) as update is not additive
        t_spk2utt = data_io.load_one_tomany(path / 'spk2utt')
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
    print(make_kaldi_ds(Path("/local_disk/arges/jduret/kaldi/egs/fabiol/v2/data/fabiol_test_no_sil")))
    print(make_kaldi_ds([Path("/local_disk/arges/jduret/kaldi/egs/fabiol/v2/data/fabiol_test_no_sil"),
                    Path("/local_disk/arges/jduret/kaldi/egs/fabiol/v2/data/fabiol_enroll_no_sil")]))