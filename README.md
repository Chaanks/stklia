# Simple-tklia

⚠️ **This is a work in progress, your feedback is welcomed** ⚠️

## Trivia

The overall pipeline for training a speaker representation network has two main components, which are referred to in this repo as a generator and a classifier. The generator is the network which actually produces the embedding:

`input (Acoustic features) -> generator (i.e. ResNet) -> embedding`  

Acting on this embedding is the classifier:

`embedding -> classifier (i.e. NN projected to num_classes with Softmax) -> class prediction`

In a classic scenario, this classifier is usually a feed forward network which projects to the number of classes, and trained using Cross Entropy Loss. This repo includes some alternate options such as angular penalty losses.

## How to install
To install the Simple-tklia toolkit, do the following steps:

0. We recomend using a conda venv : https://anaconda.org/anaconda/conda
1. Install PyTorch (http://pytorch.org/).
2. Clone the Simple-tklia repository:
```sh
git clone https://github.com/Chaanks/stklia
```
3.  Go into the project folder and Install the needed packages with:
```sh
pip install -r requirements.txt
```

## Data

To use this toolkit, please prepare your data with [kaldi](https://kaldi-asr.org). When specifying a kaldi dataset folder to our toolkit, please ensure that this folder contains thes files `feats.scp`, `utt2spk`, `spk2utt`.

## How to run
### Train a model
The training is handled with train_resnet.py. The script is run with a .cfg file as input like so:

```sh
python run.py --mode train --cfg config/example_speaker.cfg
```

In order to resume an experiment from an existing checkpoint interval:

```sh
python run.py --mode train --cfg config/example_speaker.cfg --resume-checkpoint 1000
```

### Test a model

```sh
python run.py --mode test --cfg config/example_speaker.cfg --checkpoint 1250
```

<!-- 
###  Extract X-Vectors

The extraction is handled within `extract_xvectors.py`. The script is run with a .cfg file as input like so:

```sh
python run.py --mode extract --cfg config/example_speaker.cfg
``` 
-->

## Configuration files
An example .cfg file for speaker training is provided below and in configs/example_speaker.cfg:


### Dataset specification

These are the locations of the datasets. Test and trial field are optional for the training. If they are not included in the config file, no evaluation is done during training.
It is possible to specify multiple folders. If so, the folder will be merged into one dataset class containing all the data.
Make sure to specify the number of features of your data with `features_per_frame`.

```ini
[Datasets]
train = path/to/kaldi/train/data/
test = path/to/kaldi/test/data/ #optional during training
    path/to/kaldi/enroll/data
trials = path/to/trials/file1    #optional during training 
    path/to/trials/file2
    path/to/trials/file3
features_per_frame = 61
```

The format of trials is as follows:

```
1 <utterance_a> <utterance_b>
0 <utterance_a> <utterance_c>
```

### Hyperparameters

Most of these configurable hyper-parameters are fairly self-explanatory.

```ini
[Hyperparams]
lr = 0.2
batch_size = 128
max_seq_len = 400
no_cuda = False
seed = 1234
num_iterations = 2000 # total num batches to train
momentum = 0.5
scheduler_steps = [1000, 1500, 1750]
scheduler_lambda = 0.5 # multiplies lr by this value at each step above
multi_gpu = False # dataparallel
```

### Model

This section is used to specify the model size, the embeddings size, pooling mode.
Pooling can be `min`, `max`, `mean`, `std`, `statistical`.

```ini
[Model]
emb_size = 256
layers = [3, 4, 6, 3]
num_filters = [32, 64, 128, 256]
zero_init_residual = True
pooling = statistical
```

### Outputs


The model_dir is the folder in which models are stored. At every checkpoint_interval iterations, both the generator and classifier will be stored as a .pt model inside this folder. Each model has the form: g_\<iterations\>.pt, c_\<iterations\>.pt. This is relevant to the above section of how to resume from a previous checkpoint. For example, to resume from the 1000th iteration, both g_1000.pt, c_1000.pt must exist in checkpoints_dir.

```ini
[Outputs]
model_dir = exp/example_exp_speaker # place where models are stored
checkpoint_interval = 10 # Interval to save models and also evaluate
checkpoints_dir = checkpoints # checkpoints will be stored in <model_dir>/<checkpoints_dir>/
log_interval = 1 
```

## Known problems

A batch size too big can lead to Cuda out of memory

# References
https://github.com/cvqluu/dropclass_speaker  
https://github.com/4uiiurz1/pytorch-adacos  
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch  
