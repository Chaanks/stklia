# Simple-tklia

⚠️ **This is a work in progress, your feedback is welcomed** ⚠️  
✅ Checkout our [trello](https://trello.com/b/MGHEfOjL) to see our current work.

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
pip install -r requirement.txt
```

## Data preparation

To use this toolkit, please prepare your data with [kaldi](https://kaldi-asr.org). When specifying a kaldi dataset folder to our toolkit, please ensure that this folder contains thes files `feats.scp`, `utt2spk`, `spk2utt`.  
Tutorials on how to prepare some popular datasets can be found [here](https://github.com/Chaanks/stklia/tree/master/recipes).

## How to run

The training and testing of a model is handled with the script `run.py` :
```sh
python run.py [-h] -m {train,test} --cfg CFG [--checkpoint CHECKPOINT]
```

### Train a model

To train a model, simply specify the train mode and a configuration file to `run.py`.  
Exemple :

```sh
python run.py --mode train --cfg config/example_speaker.cfg
```

In order to resume an experiment from an existing checkpoint interval, add the `--checkpoint` arguement.  
Exemple:

```sh
python run.py --mode train --cfg config/example_speaker.cfg --checkpoint 1000
```

### Test a model

To test a model, simply specify the test mode and a configuration file to `run.py`.  
Exemple :
```sh
python run.py --mode test --cfg config/example_speaker.cfg
```

A checkpoint can be added to the argument to test. If no checkpoint is specified, the last iteration will be used.
Exemple :
```sh
python run.py --mode test --cfg config/example_speaker.cfg --checkpoint 1250
```

### Tensorboard

Training and validation are saved in a tensorboard in the folder `stklia/runs/`. To visualize the data, use the command :
```bash
tensorboard --logdir runs/
```

If launching tensorboard from a remote server, use the `--port <your_port>`and `--bind_all` options :
```bash
tensorboard --logdir runs/ --port <your_port> --bind_all
```

> Note: make sure to be in the conda venv or to have tensorboard installed

###  Extract X-Vectors

To extract the x-vector of a dataset, use the extract.py script with the following command :
```sh
python extract.py [-h] -m MODELDIR [--checkpoint CHECKPOINT] -d DATA [-f {ark,txt,pt}] 
``` 

 - `--modeldir` should be the path to the trained model you want to extract the xvectors with. This folder should at least contain de `checkpoints` folder and the `experiment_settings.cfg` file. The `experiment_settings.cfg` (or `experiment_settings_resume.cfg` if present) will be used to create the generator.
 - `--checkpoint` Is an optional parameter, it can be used to specify a checkpoint for the extraction. If not specifed the last checkpoint will be used.
 - `--data` can be used in 2 manners : You can specify a kaldi folder, and the folder's data will be extracted. Or, you can simply specify test/eval/train, and the corresponding dataset of `experiment_settings.cfg` will be extracted.
 - `--format` is used to specify the output format of the xvectors. It can be kaldi (`ark`), text (`txt`), or pytorch (`pt`). Default is kaldi.

## Configuration files
An example .cfg file for speaker training is provided below and in configs/example_speaker.cfg:


### Dataset specification

These are the locations of the datasets.
`train` field is mandatory for train mode.  
`test` and `test_trial` are mandatory for test mode.  
`eval` and `eval_trial` are optional field. If they are not specified, no evaluation is done during training.  
It is possible to specify multiple folders. If so, the folder will be merged into one dataset class containing all the data.
Make sure to specify the number of features of your data with `features_per_frame`.

```ini
[Datasets]
train = path/to/kaldi/train/data/

eval = path/to/kaldi/eval/data/
eval_trial = path/to/trials/file1

test = path/to/kaldi/test/data/
    path/to/kaldi/enroll/data
test_trial = path/to/trials/file1
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

## Slurm recommendation

With a batch size of `128`, since batch size too big can lead to Cuda out of memory.

```sh
#!/bin/bash
#SBATCH --job-name="👾"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --mem=16G
#SBATCH --time=120:00:00

source /etc/profile.d/conda.sh
source /etc/profile.d/cuda.sh

conda activate <venv name>

python run.py ...

```

# References
https://github.com/cvqluu/dropclass_speaker  
https://github.com/4uiiurz1/pytorch-adacos  
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch  
