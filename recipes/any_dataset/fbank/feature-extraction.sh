#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.

# Création des liens symbolique vers les dossiers de scripts

nj=8
fbank_config=conf/fbank.conf
data_in=data/train
data_out=data/train_no_sil
features_out=features
kaldi_root=/home/$USER/kaldi/
help_message="Usage: $0 [options]
Options:
    --data-folder <data folder name>    # the name of the data folder
    --nj <nj>                           # number of parallel jobs.
Note: 
    <data-folder> defaults to '$data_in', meaning the script will process the data located in data/train/
    <nj> defaults to $nj
Exemple:
     ./feature-extraction.sh --nj 8 --data-in data/train --data-out data/train-no-sil --features-out features/

"

. utils/parse_options.sh

export KALDI_ROOT=$kaldi_root
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# Calcul des fbanks
steps/make_fbank.sh --write-utt2num-frames true --fbank-config $fbank_config --nj $nj --cmd "run.pl" ${data_in} exp/make_fbank ${features_out}
utils/fix_data_dir.sh ${data_in}

# Calcul des moments de silence
sid/compute_vad_decision.sh --nj $nj --cmd "run.pl" ${data_in} exp/make_vad `pwd`/vad
utils/fix_data_dir.sh ${data_in}

# This script applies CMVN and removes nonspeech frames.
local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "run.pl" ${data_in} ${data_out} exp/$(basename ${data_out})
utils/fix_data_dir.sh ${data_out}
