import argparse
from pathlib import Path

import numpy as np
from torchsummary import summary
from models import resnet, XTDNN, LightCNN
from parser import fetch_config


# ARGUMENTS PARSING
parser = argparse.ArgumentParser(description='Train and test of ResNet for speaker verification')

parser.add_argument('--cfg', type=str, required=True, help="Path to a config file")
args = parser.parse_args()

# Check that the config file exist
args.cfg = Path(args.cfg)
assert args.cfg.is_file(), f"No such file {args.cfg}"

# CONFIG FILE PARSING
args = fetch_config(args, 1)

if args.model == 'RESNET':
    generator = resnet(args)
elif args.model == 'XTDNN':
    generator = XTDNN()
elif args.model == 'CNN':
    generator = LightCNN()

generator = generator.cuda()
summary(generator, (1, 400, 61))


model_p = filter(lambda p: p.requires_grad, generator.parameters())
params = sum([np.prod(p.size()) for p in generator.parameters()])
print(params)