## Prerequesites

1. Install Kaldi (http://kaldi-asr.org/) and  add the path of the Kaldi binaries into `$HOME/.bashrc`. For instance, make sure that .bashrc contains the following paths:
```sh
export KALDI_ROOT=/PATH/TO/KALDI
export PATH=\
${KALDI_ROOT}/tools/openfst:\
${KALDI_ROOT}/src/bin:\
${KALDI_ROOT}/src/chainbin:\
${KALDI_ROOT}/src/featbin:\
${KALDI_ROOT}/src/gmmbin:\
${KALDI_ROOT}/src/ivectorbin:\
${KALDI_ROOT}/src/nnet3bin:\
${KALDI_ROOT}/src/nnetbin:\
$PATH
```
You can use the script in the `utils` folder. Remember to change the KALDI_ROOT variable using your path.
```sh
source ../utils/kaldi_path.sh
```
> **Note:**  
> As a first test to check the installation, open a bash shell, type "copy-feats" or > >> "copy-vector" and make sure no errors appear.

## Voxceleb tutorial

### Data Preparation
The speakers dataset used here are VoxCeleb (1+2). The only portion used for training is VoxCeleb 2 (train portion). The evaluation is computed on VoxCeleb 1 (train+test portion).

### VoxCeleb
We use the VoxCeleb recipe in Kaldi to create our training examples.
The modified version is found in `run_vc_dataprep.sh`

To run this, modify the variables at the top of the file to point to the location of VoxCeleb 1, 2, and MUSAN corpora, and then run the following, with `$KALDI_ROOT` referring to the location of your Kaldi installation.

```sh
mv run_vc_dataprep.sh $KALDI_ROOT/egs/voxceleb/v2/
cd $KALDI_ROOT/egs/voxceleb/v2
source path.sh
./run_vc_dataprep.sh
```


Running this dataprep recipe does the following:

* Makes Kaldi data folder for VoxCeleb 2 (just train portion)
* Makes Kaldi data folder for VoxCeleb 1 (train+test portion)
* Makes MFCCs for each dataset
* Augments the VoxCeleb 2 train portion with MUSAN and RIR_NOISES data, in addition to removing silence frames.
* Removes silence frames from VoxCeleb 1

> **Note**:  
> The resulting train dataset should have 5994 speakers (5994 lines in > spk2utt).


### Additional necessary data prep
For speaker datasets intended to be used as evaluation/test datasets, there must also be a file called veri_pairs within these data folders. This is similar to a trials file used by Kaldi which lists the pairs of utterances that are to be compared, along with the true label of whether or not they belong to the same speaker.

The format of this veri_pairs file is as follows:
```
1 <utterance_a> <utterance_b>
0 <utterance_a> <utterance_c>
```

where 1 indicates both utterances have the same identity, and 0 indicates a different identity. To obtain the primary verification list for VoxCeleb, the following code can be run:

```sh
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
python ../utils/vctxt_to_veri_pairs.py -i veri_test.txt -o $KALDI_ROOT/egs/voxceleb/v2/data/voxceleb1_nosil/veri_pairs
rm veri_text.txt
```

### Configuration files

```ini
[Datasets]
train = $KALDI_ROOT/egs/voxceleb/v2/data/train_combined_no_sil
eval = $KALDI_ROOT/egs/voxceleb/v2/data/voxceleb1_nosil #OPTIONAL
eval_trials = $KALDI_ROOT/egs/voxceleb/v2/data/voxceleb1_nosil/veri_pairs #OPTIONAL
features_per_frame = 30
```


### Train the model

```sh
python run.py --mode train --cfg config/<config_name>.cfg
```
