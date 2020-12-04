
#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.

. ./path.sh
set -e

nj=8
fbank_config=conf/fbank.conf
data_folder=train
musan_root=musan
min_len=400
help_message="Usage: $0 [options]
Options:
    --data-folder <data folder name>    # the name of the data folder
    --nj <nj>                           # number of parallel jobs.
Note: <data-folder> defaults to '$data_folder', meaning the script will process the data located in data/train/
      <nj> defaults to $nj
"

frame_shift=0.01
awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/${data_folder}/utt2num_frames > data/${data_folder}/reco2dur

if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
fi

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

# Make a reverberated version of the data.  Note that we don't add any
# additive noise here.
steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/${data_folder} data/${data_folder}_reverb
cp data/${data_folder}/vad.scp data/${data_folder}_reverb/
utils/copy_data_dir.sh --utt-suffix "-reverb" data/${data_folder}_reverb data/${data_folder}_reverb.new
rm -rf data/${data_folder}_reverb
mv data/${data_folder}_reverb.new data/${data_folder}_reverb

# Prepare the MUSAN corpus, which consists of music, speech, and noise
# suitable for augmentation.
steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

# Get the duration of the MUSAN recordings.  This will be used by the
# script augment_data_dir.py.
for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
done

# Augment with musan_noise
steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${data_folder} data/${data_folder}_noise
# Augment with musan_music
steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${data_folder} data/${data_folder}_music
# Augment with musan_speech
steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${data_folder} data/${data_folder}_babble

# Combine reverb, noise, music, and babble into one directory.
utils/combine_data.sh data/${data_folder}_noise data/${data_folder}_reverb data/${data_folder}_noise data/${data_folder}_music data/${data_folder}_babble

# Make Fbank for the augmented data.  Note that we do not compute a new
# vad.scp file here.  Instead, we use the vad.scp from the clean version of
# the list.
steps/make_fbank.sh --fbank-config $fbank_config --nj $nj --cmd "run.pl" data/${data_folder}_noise exp/make_fbank `pwd`/fbank

# Combine the clean and augmented VoxCeleb2 list.  This is now roughly
# double the size of the original clean list.
utils/combine_data.sh data/${data_folder}_aug data/${data_folder}_noise data/${data_folder}
data_folder=${data_folder}_aug

# This script applies CMVN and removes nonspeech frames.
local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "run.pl" data/${data_folder} data/${data_folder}_no_sil exp/${data_folder}_no_sil
utils/fix_data_dir.sh data/${data_folder}_no_sil

# Removes the utt that are shorter than min_len
mv data/${data_folder}_no_sil/utt2num_frames data/${data_folder}_no_sil/utt2num_frames.bak
awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/${data_folder}_no_sil/utt2num_frames.bak > data/${data_folder}_no_sil/utt2num_frames
utils/filter_scp.pl data/${data_folder}_no_sil/utt2num_frames data/${data_folder}_no_sil/utt2spk > data/${data_folder}_no_sil/utt2spk.new
mv data/${data_folder}_no_sil/utt2spk.new data/${data_folder}_no_sil/utt2spk
utils/fix_data_dir.sh data/${data_folder}_no_sil
