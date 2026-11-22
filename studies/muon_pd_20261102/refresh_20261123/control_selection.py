"""Score the existing equal-compute ordinary checkpoints on the development banks."""
import os
from pathlib import Path
import subprocess
import sys
from odd_trial import HERE, base
import json


def read(path):
    return json.loads(path.read_text())


def run(prep, output):
    output.mkdir(parents=True, exist_ok=True)
    os.nice(10)
    rows = []
    for endpoint in sorted((prep/'report4-equal-compute-20261122/trained-endpoints').glob('*.json')):
        record = read(endpoint)
        folder = Path(record['folder'])
        cfg = read(folder/'config.json')
        cell = cfg['cell']
        assert cfg['seeds'] == [0] and record['checkpoint_sha256'] == base.digest(record['endpoint'])
        ds = cell['dataset']
        dest = output / cell['id']
        if not (dest/'COMPLETE.json').exists():
            subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--folder',str(folder),'--data-root',
                            str(prep/'data/synthetic'),'--assets',str(prep/'coefficient-20261116/assets'),
                            '--output',str(dest),'--split','selection','--steps',str(record['target']),
                            '--device','cpu'],check=True)
        done=read(dest/'COMPLETE.json')
        assert done['passed'] and not done['witness'] and done['steps']==[record['target']]
        score, = read(dest/'summary.json')
        assert score['checkpoint_sha256'] == record['checkpoint_sha256']
        binding=read(dest/'binding.json')
        old, = (prep/'coefficient-20261116/analysis/selection').glob(cell['id']+'/summary.json')
        ref, = [r for r in read(old) if r['step']==20000]
        assert read(old.parent/'binding.json')['assets']['sha256']==binding['assets']['sha256']
        rows.append(dict(dataset=ds,family=cell['family'],matched_steps=score['step'],
                         matched_sw2=score['pooled_sw2'],original_20k_sw2=ref['pooled_sw2'],
                         matched_checkpoint_sha256=score['checkpoint_sha256'],
                         original_checkpoint_sha256=ref['checkpoint_sha256'],
                         assets_sha256=binding['assets']['sha256']))
        base.dump(output/'scores.json',rows)
    assert len(rows)==4


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
