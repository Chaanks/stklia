
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
from models import resnet, NeuralNetAMSM, XTDNN, LightCNN
from cuda_test import cuda_test, get_device

if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Extract the xvectors of a dataset given a model. (xvectors will be extracted in <model_dir>/xvectors/<checkpoint>/)')

    parser.add_argument("-m", "--model_dir", type=str, required=True, help="The path to the model directory you want to extract the xvectors with. This dir should containt the file experiement.cfg")
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='The path where extract the xvectors')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='The path were datataset is located')
    parser.add_argument('--checkpoint', '--resume-checkpoint', type=int, default=-1,
                            help="Model Checkpoint to use. Default (-1) : use the last one ")
    parser.add_argument("-f", "--format", type=str, choices=["ark", "txt", "pt"], default="ark", help="The output format you want the x-vectors in.")
    args = parser.parse_args()
    args.model_dir = Path(args.model_dir)

    # Check that de cfg file exist
    set_res_file = Path(args.model_dir) / "experiment_settings_resume.cfg"
    args.cfg = set_res_file if set_res_file.is_file() else Path(args.model_dir) / "experiment_settings.cfg"
    assert args.cfg.is_file(), f"No such file {args.cfg}"

    args = fetch_config(args)

    # Check that the output folder exist
    args.out_dir = Path(args.out_dir)
    assert args.out_dir.is_dir(), f"No such directory {args.out_dir}"
    
    # Check that the dataset path exist
    args.data_dir = Path(args.data_dir)
    assert args.data_dir.is_dir(), f"No such directory {args.out_dir}"
    ds_extract = dataset.make_kaldi_ds_eval(args.data_dir, seq_len=5000, evaluation=True)
    print(ds_extract)

    args.out_dir = Path(args.out_dir) / args.model_dir.stem / args.data_dir.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    # Generator and classifier definition
    if args.model == 'RESNET':
        generator = resnet(args)
    elif args.model == 'XTDNN':
        generator = XTDNN()
    elif args.model == 'CNN':
        generator = LightCNN()

    #generator = resnet(args)

    generator.load_state_dict(torch.load(g_path), strict=False)
    generator = generator.to(device)
    generator.eval()

    # Extract xv
    if args.format in ["ark", "txt"]:
        if args.format == "ark":
            ark_scp_xvector = f'ark:| copy-vector ark:- ark,scp:{args.out_dir}/xvectors.ark,{args.out_dir}/xvectors.scp'
            mode = "wb"
        if args.format == "txt":
            ark_scp_xvector = f'ark:| copy-vector ark:- ark,t:{args.out_dir}/xvectors.txt'
            mode = "w"

        with kaldi_io.open_or_fd(ark_scp_xvector, mode) as f:
            with torch.no_grad():
                for i in tqdm(range(len(ds_extract))):
                    feats, utt = ds_extract.__getitem__(i)
                    feats = feats.unsqueeze(0).unsqueeze(1).to(device)
                    embeds = generator(feats).cpu().numpy()
                    kaldi_io.write_vec_flt(f, embeds[0], key=utt)
    
