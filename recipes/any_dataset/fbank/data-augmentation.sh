#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2020   Vincent Brignatz
# Apache 2.0.


nj=8
musan_root=musan
rirs_root=RIRS_NOISES
min_len=400
fbank_config=conf/fbank.conf
data_in=data/train
features_out=feats/
exp_out=log
kaldi_root=/home/$USER/kaldi/
help_message="Usage: $0 [options]
Options:
    --nj <nj>      # number of parallel jobs.
    --data-in      # the path of the data input folder
    --features-out # the path of the feature output folder (can be heavy)
    --kaldi-root   # the path to the kaldi install folder 
    --rirs-root    # the path to the the rirs noises dataset (https://www.openslr.org/28/)
    --musan-root   # the path to the musan dataset (http://www.openslr.org/17/)
Exemple:
    ./data-augmentation.sh --nj 8 --data-in data/train --features-out feats/ --kaldi-root /home/me/kaldi --rirs-root RIRS_NOISES --musan-root musan
    will procude the folder data/train_combined_no_sil
"

# Parse command-line options :
. utils/parse_options.sh

# Create the file path.sh need by all the kaldi scripts :
echo "export KALDI_ROOT=$kaldi_root
export PATH=\$PWD/utils/:\$KALDI_ROOT/tools/openfst/bin:\$KALDI_ROOT/tools/sph2pipe_v2.5:\$PWD:\$PATH
[ ! -f \$KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 \"The standard file \$KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!\" && exit 1
. \$KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C" > path.sh
. ./path.sh

frame_shift=0.01
awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' ${data_in}/utt2num_frames > ${data_in}/reco2dur

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, $rirs_root/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, $rirs_root/simulated_rirs/mediumroom/rir_list")

# Make a reverberated version of the data.  Note that we don't add any
# additive noise here.
steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    ${data_in} ${data_in}_reverb
cp ${data_in}/vad.scp ${data_in}_reverb/
utils/copy_data_dir.sh --utt-suffix "-reverb" ${data_in}_reverb ${data_in}_reverb.new
rm -rf ${data_in}_reverb
mv ${data_in}_reverb.new ${data_in}_reverb

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
steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" ${data_in} ${data_in}_noise
# Augment with musan_music
steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" ${data_in} ${data_in}_music
# Augment with musan_speech
steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" ${data_in} ${data_in}_babble

# Combine reverb, noise, music, and babble into one directory.
utils/combine_data.sh ${data_in}_aug ${data_in}_reverb ${data_in}_noise ${data_in}_music ${data_in}_babble

# Make Fbank for the augmented data.  Note that we do not compute a new
# vad.scp file here.  Instead, we use the vad.scp from the clean version of
# the list.
steps/make_fbank.sh --fbank-config $fbank_config --nj $nj --cmd "run.pl" ${data_in}_aug exp/make_fbank $features_out/$(basename ${data_in})_aug
utils/fix_data_dir.sh ${data_in}_aug

# Combine the clean and augmented VoxCeleb2 list.  This is now roughly
# double the size of the original clean list.
utils/combine_data.sh ${data_in}_combined ${data_in}_aug ${data_in}

# This script applies CMVN and removes nonspeech frames.
data_out=${data_in}_combined_no_sil
local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "run.pl" ${data_in}_combined $data_out $features_out/$(basename ${data_in})_combined_no_sil
utils/fix_data_dir.sh $data_out

# Removes the utt that are shorter than min_len
mv $data_out/utt2num_frames $data_out/utt2num_frames.bak
awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data_out/utt2num_frames.bak > $data_out/utt2num_frames
utils/filter_scp.pl $data_out/utt2num_frames $data_out/utt2spk > $data_out/utt2spk.new
mv $data_out/utt2spk.new $data_out/utt2spk
utils/fix_data_dir.sh $data_out
