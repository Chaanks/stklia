#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.

# Création des liens symbolique vers les dossiers de scripts

. ./path.sh
set -e

nj=8
fbank_config=conf/fbank.conf
data_folder=train
help_message="Usage: $0 [options]
Options:
    --data-folder <data folder name>    # the name of the data folder
    --nj <nj>                           # number of parallel jobs.
Note: <data-folder> defaults to '$data_folder', meaning the script will process the data located in data/train/
      <nj> defaults to $nj
"

. utils/parse_options.sh

# Calcul des fbanks
steps/make_fbank.sh --write-utt2num-frames true --fbank-config $fbank_config --nj $nj --cmd "run.pl" data/${data_folder} exp/make_fbank `pwd`/fbank
utils/fix_data_dir.sh data/${data_folder}

# Calcul des moments de silence
sid/compute_vad_decision.sh --nj $nj --cmd "run.pl" data/${data_folder} exp/make_vad `pwd`/fbank
utils/fix_data_dir.sh data/${data_folder}

# This script applies CMVN and removes nonspeech frames.
local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "run.pl" data/${data_folder} data/${data_folder}_no_sil exp/${data_folder}_no_sil
utils/fix_data_dir.sh data/${data_folder}_no_sil
