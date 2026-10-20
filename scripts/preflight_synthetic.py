"""Read-only checks to run before the user approves sbatch submission."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic.grid import grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/synthetic/train_v1.json')
    p.add_argument('--data-root',default='data/synthetic/v1')
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    root=Path(args.data_root)
    manifest=json.loads((root/'manifest.json').read_text())
    if cfg['steps']<=0 or cfg['eval_interval']<=0 or cfg['sample_interval']<=0 or cfg['checkpoint_interval']<=0:
        raise ValueError('nonpositive training intervals')
    entries = grid(args.config)
    if cfg['seeds'] != [0] or not entries or len(set(entries)) != len(entries):
        raise ValueError('expected a nonempty unique grid using only seed 0')
    for dataset in cfg['datasets']:
        if dataset not in manifest['datasets']:
            raise ValueError(f'missing dataset {dataset}')
        spec=manifest['datasets'][dataset]
        for split in ('train','validation','test','probe'):
            relative=f'{dataset}/{split}.npz'
            file=root/relative
            if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=manifest['files'][relative]['sha256']:
                raise ValueError(f'missing or corrupt: {file}')
        if spec['dim'] != 2:
            raise ValueError('v1 grid is for 2D only')
    for model in cfg['models']:
        command=[sys.executable,'-m','synthetic.train','--config',args.config,
                 '--dataset',cfg['datasets'][0],'--model',model,'--seed',str(cfg['seeds'][0]),
                 '--device','cpu','--dry-run']
        subprocess.run(command,check=True)
    print(f'Preflight OK: {len(entries)} array entries, {len(cfg["models"])} models checked on CPU, {len(cfg["datasets"])} datasets hashed.')


if __name__=='__main__':
    main()
