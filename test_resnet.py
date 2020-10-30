#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_resnet.py: This file contains function to score 
a trained resnet on various trials.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import torch
import pickle
import sys
import numpy as np

from collections import OrderedDict
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm
from math import log10, floor
from pathlib import Path
from loguru import logger

import dataset
import data_io
from models import resnet34
from parser import fetch_args_and_config
from cuda_test import get_device

@logger.catch
def compute_spk_xvec(generator, ds, device):
    """
        Extract all the x-vectors of a speaker
        and calc the mean to get the x-vector 
        representation of the speaker.
    """
    # set the model in eval mode
    generator.eval()

    spk_sum = {}
    spk_count = {}

    with torch.no_grad(): # reduce memory usage
        for i in tqdm(range(len(ds))):
            feats, spk, utt = ds.__getitem__(i) # TODO: replace by ds[i] ?
            feats = feats.unsqueeze(0).unsqueeze(1).to(device)
            embeds = generator(feats).cpu().numpy()
            spk = ds.label_enc.inverse_transform([spk])[0].item() # BUG: second definition of var spk

            if spk not in spk_sum:
                spk_sum[spk] = embeds[0]
                spk_count[spk] = 1
            else:
                spk_sum[spk] = spk_sum[spk] + embeds[0]
                spk_count[spk] += 1

    # Calculate the mean for each speaker
    spks_mean = {}
    for spk in spk_sum.keys():
        spks_mean[spk] = spk_sum[spk] / spk_count[spk]

    # Rturn the spk xvec and the spk list
    return list(spks_mean.values()), list(spks_mean.keys())

@logger.catch
def compute_utt_xvec(generator, ds, device):
    """
        Extract all the x-vectors of all the sessions.
    """
    # set the model in eval mode
    generator.eval()

    all_embeds = {}

    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            feats, spk, utt = ds.__getitem__(i)
            feats = feats.unsqueeze(0).to(device)
            feats = feats.unsqueeze(1)
            embeds = generator(feats).cpu().numpy()
            all_embeds[utt] = embeds
    
    return list(all_embeds.values()), list(all_embeds.keys())

@logger.catch
def compute_unique_utt_xvec(generator, ds, trial, device):
    """
        TODO Extract the x-vectors only for sessions required by trial.
    """
    # set the model in eval mode
    generator.eval()

    veri_labs, veri_0, veri_1 = data_io.load_n_col(trial)
    veri_0 = list(filter(lambda utt: utt in ds.utt2spk, veri_0))
    veri_1 = list(filter(lambda utt: utt in ds.utt2spk, veri_1))
    veri_utts = list(set(np.concatenate([veri_0, veri_1])))

    all_embeds = {}

    with torch.no_grad():
        for i in tqdm(range(len(veri_utts))):
            feats = ds.get_utt_feats(veri_utts[i])
            feats = feats.unsqueeze(0).to(device)
            feats = feats.unsqueeze(1)
            embeds = generator(feats).cpu().numpy()
            all_embeds[veri_utts[i]] = embeds
    
    return list(all_embeds.values()), list(all_embeds.keys())

def eer_from_ers(fpr, tpr):
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer
    # return fpr[np.nanargmin(np.absolute((1 - tpr - fpr)))]

def scores_from_pairs(vecs0, vecs1):
    return paired_distances(vecs0, vecs1, metric='cosine')

def compute_min_dcf(fpr, tpr, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    # adapted from compute_min_dcf.py in kaldi sid
    # thresholds, fpr, tpr = list(zip(*sorted(zip(thresholds, fpr, tpr))))
    incr_score_indices = np.argsort(thresholds, kind="mergesort")
    thresholds = thresholds[incr_score_indices]
    fpr = fpr[incr_score_indices]
    tpr = tpr[incr_score_indices]

    fnr = 1. - tpr
    min_c_det = float("inf")
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf

def score_utt_utt(generator, ds_test, device, mindcf=False):
    """ 
        Score the model on the trials of type :
        <spk> <utt> 0/1
    """
    trials = ds_test.trials
    if not isinstance(trials, list):
        trials = [trials]

    #all_embeds, all_utts = compute_utt_xvec(generator, ds_test, device)

    all_res = {}
    for verilist_path in trials:
        assert verilist_path.is_file()
        all_embeds, all_utts = compute_unique_utt_xvec(generator, ds_test, verilist_path, device)

        veri_labs, veri_utt1, veri_utt2 = data_io.load_n_col(verilist_path)
        veri_labs = np.asarray(veri_labs, dtype=int)

        all_embeds = np.vstack(all_embeds)
        all_embeds = normalize(all_embeds, axis=1)
        all_utts = np.array(all_utts)

        utt_embed = OrderedDict({k:v for k, v in zip(all_utts, all_embeds)})

        emb0 = np.array([utt_embed[k] for k in veri_utt1])
        emb1 = np.array([utt_embed[k] for k in veri_utt2])

        scores = scores_from_pairs(emb0, emb1)
        fpr, tpr, thresholds = roc_curve(1 - veri_labs, scores, pos_label=1, drop_intermediate=False)
        eer = eer_from_ers(fpr, tpr)

        if mindcf:
            mindcf1 = compute_min_dcf(fpr, tpr, thresholds, p_target=0.01)
            mindcf2 = compute_min_dcf(fpr, tpr, thresholds, p_target=0.001)
            print(f'[{verilist_path.name}] EER :{eer*100:.4f}  minDFC p=0.01 :{mindcf1}  minDFC p=0.001 :{mindcf2}  ')
            all_res[verilist_path.name] = {"eer":eer, "mindcf1":mindcf1, "mindcf2":mindcf2}
        else:
            print(f'[{verilist_path.name}] EER :{eer*100:.4f}')
            all_res[verilist_path.name] = {"eer":eer}
    return all_res

def score_spk_utt(generator, ds_enroll, ds_test, trials, device):
    """ 
        Score the model on the trials of type :
        <spk> <utt> 0/1
    """
    if not isinstance(trials, list):
        trials = [trials]

    # Get enroll embeddings
    spks_mean, spks = compute_spk_xvec(model, ds_enroll, device)

    # Get test embeddings
    utts_embeds, utts = compute_utt_xvec(model, ds_test, device)

    all_eer = {}
    for verilist_path in trials:
        assert verilist_path.is_file()

        veri_labs, veri_spk, veri_utt = dataset.load_n_col(verilist_path)
        veri_labs = np.asarray(veri_labs, dtype=int)

        all_embeds = spks_mean + utts_embeds
        all_embeds = np.vstack(all_embeds)
        all_embeds = normalize(all_embeds, axis=1)

        all_keys = spks + utts
        all_keys = np.array(all_keys)

        key_embed = OrderedDict({k:v for k, v in zip(all_keys, all_embeds)})

        emb0 = np.array([key_embed[k] for k in veri_spk])
        emb1 = np.array([key_embed[k] for k in veri_utt])

        scores = scores_from_pairs(emb0, emb1)
        fpr, tpr, thresholds = roc_curve(1 - veri_labs, scores, pos_label=1, drop_intermediate=False)
        eer = eer_from_ers(fpr, tpr)

        print(f'[{verilist_path.name}] EER :{eer*100:.4f}')
        all_eer[verilist_path.name] = {"eer":eer}
    return all_eer
