"""Map Slurm array IDs to the reviewed synthetic experiment grid."""
import argparse
import itertools
import json
from pathlib import Path


def grid(config_path):
    cfg=json.loads(Path(config_path).read_text())
    return list(itertools.product(cfg['datasets'],cfg['models'],cfg['seeds']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/synthetic/train_v1.json')
    p.add_argument('--index',type=int)
    args=p.parse_args()
    combinations=grid(args.config)
    if args.index is None:
        for i,entry in enumerate(combinations):
            print(i,*entry)
    else:
        if not 0<=args.index<len(combinations):
            p.error(f'array index must be 0..{len(combinations)-1}')
        print(*combinations[args.index])
