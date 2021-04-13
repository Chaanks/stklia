import torch
import argparse
from numpy import array_str

from pathlib import Path
from subprocess import run
from os import chdir, getcwd

import dataset
from parser import fetch_config
from models import resnet34
from cuda_test import cuda_test, get_device

""" wav2xv.py : extrait le xvecteur d'un fichier wav.
usage: wav2xv.py [-h] --model MODEL [--checkpoint CHECKPOINT] [--output OUTPUT] --kaldi KALDI
                 wav [wav ...]

Extract the xvectors of a file.

positional arguments:
  wav                   The wav file(s) to extract the x-vector from.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The model directory to use. should contain expirement_settings.cfg
  --checkpoint CHECKPOINT, --resume-checkpoint CHECKPOINT
                        Model Checkpoint to use. Default (-1) : use the last one
  --output OUTPUT       The output file you want the x-vector in.
  --kaldi KALDI         The path to the kaldi installation.
"""

# --model MODEL           /local_disk/arges/vbrignatz/tklia/exp/multilang_std
# --wav WAV               vincent_josse_20652055.wav
# --output OUTPUT         xvectors
# --kaldi KALDI           /local_disk/arges/jduret/kaldi
# wav2xv.py $(ls *.wav) --model /local_disk/arges/vbrignatz/tklia/exp/multilang_std --kaldi /local_disk/arges/jduret/kaldi

if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Extract the xvectors of a file.')

    parser.add_argument('wav', type=str, nargs='+',
        help="The wav file(s) to extract the x-vector from.")
    parser.add_argument("--model", type=str, required=True,
        help="The model directory to use. should contain expirement_settings.cfg")
    parser.add_argument('--checkpoint', '--resume-checkpoint', type=int, default=-1,
        help="Model Checkpoint to use. Default (-1) : use the last one ")
    parser.add_argument('--output', type=str, default='xvector.txt',
        help="The output file you want the x-vector in.")
    parser.add_argument('--kaldi', type=str, required=True,
        help="The path to the kaldi installation.")
    parser.add_argument('--tmp', type=str, default='.tmp/',
        help="The temporary directory")
    # parser.add_argument("-f", "--format", type=str, choices=["ark", "txt"], default="txt",
    #   help="The output format you want the x-vectors in.")

    args = parser.parse_args()


    args.model  = Path(args.model)
    args.output = Path(args.output)
    args.kaldi  = Path(args.kaldi)
    args.wav    = [ Path(f) for f in args.wav ]

    # Create .tmp dir and subdirectories
    print("Creating tmp folders")
    tmp_folder = Path(args.tmp)
    tmp_folder.mkdir(parents=True, exist_ok=True)

    data_folder = tmp_folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)

    feature_folder = tmp_folder / "feats"
    feature_folder.mkdir(parents=True, exist_ok=True)

    exp_folder = tmp_folder / "exp"
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Create wav.scp, spk2utt, utt2spk files in .tmp/data/ dir

    # Note : If we dont know the id of the speakers, kaldi recommend 
    # making the spk id the same as the utt id. See http://kaldi-asr.org/doc/data_prep.html
    # and search for word bold

    print("Creating kaldi metadata")
    
    with open(data_folder / "wav.scp", "w") as f:
        # note that kaldi needs the full path to the wav
        for wav in args.wav:
            f.write(f"{wav.stem} {wav.absolute()}\n")

    with open(data_folder / "spk2utt", "w") as f:
        for wav in args.wav:
            f.write(f"{wav.stem} {wav.stem}\n")
    
    with open(data_folder / "utt2spk", "w") as f:
        for wav in args.wav:
            f.write(f"{wav.stem} {wav.stem}\n")

    # Call feature-extraction on .tmp/data/
    cmd = f'./feature-extraction.sh --nj 1 --data-in {data_folder.absolute()} --features-out {feature_folder.absolute()} --kaldi-root {args.kaldi.absolute()} --vad-out {vad_folder.absolute()} --exp-out {exp_folder.absolute()}'
    print(f"running :\n{cmd}")

    # Since the script feature-extraction.sh works in relative paths,
    # we change the cwd to match the script directory
    working_dir = getcwd()
    chdir('recipes/any_dataset/fbank/')

    run(cmd.split(' '))
    chdir(working_dir)

    # Load dataset from .tmp/egs
    print("Generating dataset object")
    ds_extract = dataset.make_kaldi_ds(Path(str(data_folder.absolute()) + "_no_sil"), seq_len=None, evaluation=True)

    # Load system
    print('Loading config fron experiment_settings.cfg')
    set_res_file = Path(args.model) / "experiment_settings_resume.cfg"
    args.cfg = set_res_file if set_res_file.is_file() else Path(args.model) / "experiment_settings.cfg"
    assert args.cfg.is_file(), f"No such file {args.cfg}. --modeldir should contain experiment_settings.cfg"

    args = fetch_config(args) 

    # cuda_test()
    device = get_device(use_cuda=True)

    if args.checkpoint < 0:
        g_path = args.model_dir / "final_g_{}.pt".format(args.num_iterations)
    else:
        print('Using checkpoint {}'.format(args.checkpoint))
        g_path = args.checkpoints_dir / "g_{}.pt".format(args.checkpoint)

    generator = resnet34(args)
    generator.load_state_dict(torch.load(g_path), strict=False)
    generator = generator.to(device)
    generator.eval()

    # Extract xv from .tmp/data into --output
    with open(args.output, 'w') as fout:
        with torch.no_grad():
            for i in range(len(ds_extract)):
                feats, _, utt = ds_extract.__getitem__(i)
                feats = feats.unsqueeze(0).unsqueeze(1).to(device)
                embeds = torch.squeeze(generator(feats)).cpu().numpy()
                fout.write(f"{utt} {array_str(embeds, max_line_width=1000000)}\n")

    # Delete folder .tmp
    run(f'rm -r {tmp_folder}'.split(' '))
