import random
import argparse
from simple_slurm import Slurm


EMOJI = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐸', '🐵', '🐔', '🐧', '🐦', '🐤', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🐛', '🦋', '🐌', '🐞', '🐜', '🦟', '🦗', '🕷', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀', '🐠']


parser = argparse.ArgumentParser(description='Process cfg file')
parser.add_argument("-c", "--cfg", type=str, required=True, help="Put this argument to run sbatch")
args = parser.parse_args()

slurm = Slurm()
slurm.add_arguments(ntasks='1')
slurm.add_arguments(cpus_per_task='8')
slurm.add_arguments(partition='gpu')
slurm.add_arguments(gpus_per_node='rtx_2080_ti:2')
#slurm.add_arguments(gpus_per_node='1')
slurm.add_arguments(job_name=random.choice(EMOJI))
slurm.add_arguments(output=r'slurm/logs/%j.out')
slurm.add_arguments(mem='16G')
slurm.add_arguments(time='192:00:00')

slurm.sbatch(f'python run.py --cfg {args.cfg} --mode train')

#slurm.sbatch('python extract.py -m exp/RESNET34_256_statistical -d /local_disk/arges/jduret/kaldi/egs/voxceleb/fbank/data/train_combined_no_sil -f ark')

