
"""
python extract.py --modeldir path --checkpoint 6969 --data path
"""

import torch
import argparse
import kaldi_io

from tqdm import tqdm
from pathlib import Path


import dataset
from parser import fetch_config
from models import resnet34
from cuda_test import cuda_test, get_device


def extract(gen, ds, device, path):
    gen.eval()

    ark_scp_xvector = f'ark:| copy-vector ark:- ark,scp:{path}/xvector.ark,{path}/xvector.scp'
    with kaldi_io.open_or_fd(ark_scp_xvector,'wb') as f:
        with torch.no_grad():
            for i in tqdm(range(len(ds))):
                feats, _, utt = ds.__getitem__(i)
                feats = feats.unsqueeze(0).unsqueeze(1).to(device)
                embeds = gen(feats).cpu().numpy()
                kaldi_io.write_vec_flt(f, embeds[0], key=utt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the xvectors of a dataset given a model. (xvectors will be extracted in <model_dir>/xvectors/<checkpoint>/)')

    parser.add_argument("-m", "--modeldir", type=str, required=True, help="The path to the model directory you want to extract the xvectors with. This dir should containt the file experiement.cfg")
    parser.add_argument('--checkpoint', '--resume-checkpoint', type=int, default=-1,
                            help="Model Checkpoint to use. Default (-1) : use the last one ")
    parser.add_argument("-d", '--data', type=str, required=True, help="Path to the kaldi data folder to extract (Should contain spk2utt, utt2spk and feats.scp).")

    args = parser.parse_args()
    args.modeldir = Path(args.modeldir)

    set_res_file = Path(args.modeldir) / "experiment_settings_resume.cfg"
    args.cfg = set_res_file if set_res_file.is_file() else Path(args.modeldir) / "experiment_settings.cfg"
    assert args.cfg.is_file(), f"No such file {args.cfg}"

    args = fetch_config(args) 

    try:
        data_path = {"train":args.train_data_path,
                     "eval" :args.eval_data_path,
                     "test" :args.test_data_path }[args.data]
    except KeyError:
        data_path = Path(args.data)

    if data_path == None:
        raise KeyError("No dataset {0} in {1} file while given {0} as --data".format(args.data, args.cfg))
    
    if isinstance(data_path, list):
        assert any(d.is_dir() for d in data_path), f"No such dir {data_path}"
        out_dir = args.model_dir.resolve() / "xvectors" / f"{args.data}_data"
    elif isinstance(data_path, Path):
        assert data_path.is_dir(), f"No such dir {data_path}"
        out_dir = args.model_dir.resolve() / "xvectors" / str(data_path.name)
    else:
        raise TypeError("This should not append, contact dev :(")

    out_dir.mkdir(parents=True, exist_ok=True)
    ds_extract = dataset.make_kaldi_ds(data_path, seq_len=None, evaluation=True)

    cuda_test()
    device = get_device(not args.no_cuda)

    # Load generator
    if args.checkpoint < 0:
        g_path = args.model_dir / "final_g_{}.pt".format(args.num_iterations)
        g_path_test = g_path
    else:
        print('use checkpoint {}'.format(args.checkpoint))
        g_path = args.checkpoints_dir / "g_{}.pt".format(args.checkpoint)
        g_path_test = g_path

    generator = resnet34(args)
    generator.load_state_dict(torch.load(g_path), strict=False)
    generator = generator.to(device)

    
    extract(generator, ds_extract, device, out_dir)
