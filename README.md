# Simple-tklia


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

## Voxceleb tutorial

## Training a Model
The training is handled within train_speaker.py. The script is run with a .cfg file as input like so:

```sh
python train_speaker.py --cfg config/example_speaker.cfg
```

In order to resume an experiment from an existing checkpoint interval:

```sh
python train_speaker.py --cfg config/example_speaker.cfg --resume-checkpoint 50000
```

When this resuming is possible and the documentation of a .cfg files will be described below.


## Configuration files

The overall pipeline for training a speaker representation network has two main components, which are referred to in this repo as a generator and a classifier. The generator is the network which actually produces the embedding:

`input (Acoustic features) -> generator (i.e. TDNNs) -> embedding`  

Acting on this embedding is the classifier:

`embedding -> classifier (i.e. NN projected to num_classes with Softmax) -> class prediction`

In a classic scenario, this classifier is usually a feed forward network which projects to the number of classes, and trained using Cross Entropy Loss. This repo includes some alternate options such as angular penalty losses.

### Speaker
An example .cfg file for speaker training is provided below and in configs/example_speaker.cfg:

```ini
[Datasets]
train = $KALDI_PATH/egs/voxceleb/v2/data/train_combined_no_sil
test = $KALDI_PATH/egs/voxceleb/v2/data/voxceleb1_nosil #OPTIONAL
```

These are the locations of the datasets. test is **OPTIONAL** field. If they are not included in the config file, no evaluation is done during training.

```ini
[Model]
#allowed model_type : ['XTDNN', 'ETDNN' 'FTDNN']
model_type = XTDNN
```

XTDNN refers to the original x-vector architecture, and is identical up until the embedding layer. ETDNN is the Extended TDNN architecture seen in more recent architectures (also up until the embedding layer). FTDNN is the Factorized TDNN x-vector arch. The models can be viewed in models_speaker.py

```ini
[Optim]
#allowed loss_type values: ['adm', 'adacos', 'softmax', 'l2softmax', 'xvec', 'arcface', 'sphereface']
loss_type = adm
#allowed smooth_type values: ['None']
label_smooth_type = None
label_smooth_prob = 0.1
```

The loss_type field dictates the architecture of the classifier network as described above.  

* l2softmax is a simple projection to the number of classes with both embeddings and weight matrices L2 normalized.
* adm is the additive margin Softmax loss/CosFace presented in
* adacos is the adaptive cosine penalty loss presented in
* xvec is a feed forward network with one hidden layer, choosing this option and XTDNN as the model type is almost identical to the original x-vector architecture  

```ini
[Hyperparams]
lr = 0.2
batch_size = 500 # must be less than num_classes
max_seq_len = 350
no_cuda = False
seed = 1234
num_iterations = 120000 # total num batches to train
momentum = 0.5
scheduler_steps = [50000, 60000, 70000, 80000, 90000, 100000, 110000]
scheduler_lambda = 0.5 # multiplies lr by this value at each step above
multi_gpu = False # dataparallel
classifier_lr_mult = 1.
log_interval = 100
```

Most of these configurable hyper-parameters are fairly self-explanatory.

>**Note**:   
>The way the Data Loader is implemented in this repo is to force each batch to have one example from each class. For each batch, batch_size number of classes is sampled and then removed from the pool of training classes, and a random single example belonging to each class is chosen to be in the batch. These classes are sampled until there are no remaining classes in the pool, at which point the pool is replenished and the already sampled classes are cycled in again. This is repeated until the end of training.

```ini
[Outputs]
model_dir = exp/example_exp_speaker # place where models are stored
checkpoint_interval = 500 # Interval to save models and also evaluate
```

The model_dir is the folder in which models are stored. At every checkpoint_interval iterations, both the generator and classifier will be stored as a .pt model inside this folder. Each model has the form: g_<iterations>.pt, c_<iterations>.pt. This is relevant to the above section of how to resume from a previous checkpoint. For example, to resume from the 1000th iteration, both g_1000.pt, c_1000.pt must exist in model_dir.

## X-Vectors Extraction

The extraction is handled within `extract_xvectors.py`. The script is run with a .cfg file as input like so:

```sh
python extract_xvectors.py --cfg config/example_speaker.cfg
```



# References
https://github.com/cvqluu/dropclass_speaker  
https://github.com/4uiiurz1/pytorch-adacos  
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch  
