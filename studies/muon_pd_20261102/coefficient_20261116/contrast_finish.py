"""Monitor registered native runs; reuse and audit eight completed evaluations."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contrast_trial import HERE, TAG, base, torch


def pairing(anchors: dict, output: Path) -> None:
    rows = []
    schedule = anchors.get('protocol') == 'schedule_order'
    families = ['state_midpoint4'] if schedule else ['state_midpoint4', 'stage_operator_midpoint4']
    extra_name = anchors.get('extra_parameter', 'stage_delta')
    for cell in anchors['new_cases']:
        folder = Path(cell['folder'])
        new_cfg = json.loads((folder/'config.json').read_text())
        references = [r for r in anchors['references'] if r['dataset'] == cell['dataset']
                      and r['family'] in families]
        assert len(references) == len(families)
        compared = []
        for ref in references:
            parent = Path(ref['folder'])
            old_cfg = json.loads((parent/'config.json').read_text())
            ignored = {'cell', 'models', 'experiment_id', 'grid_sha256'}
            assert {k: v for k, v in new_cfg.items() if k not in ignored} == {
                k: v for k, v in old_cfg.items() if k not in ignored}
            steps = []
            for step in [0, 15000, 20000]:
                new_file = folder/'checkpoints'/f'step_{step:07d}.pt'
                old_file = parent/'checkpoints'/f'step_{step:07d}.pt'
                new = torch.load(new_file, weights_only=False, map_location='cpu')
                old = torch.load(old_file, weights_only=False, map_location='cpu')
                assert new['step'] == old['step'] == step
                assert new['dataset_manifest_sha256'] == old['dataset_manifest_sha256']
                assert torch.equal(new['data_rng'], old['data_rng'])
                if step == 0:
                    for name, value in old['model'].items():
                        assert torch.equal(value, new['model'][name]), name
                    extra = set(new['model'])-set(old['model'])
                    assert all(n.endswith(extra_name) and torch.count_nonzero(new['model'][n]) == 0 for n in extra)
                steps.append(dict(step=step, exact_data_rng=True,
                    new_checkpoint_sha256=base.digest(new_file), reference_checkpoint_sha256=base.digest(old_file)))
            compared.append(dict(family=ref['family'], recipe_paired=True, checkpoints=steps))
        rows.append(dict(id=cell['id'], dataset=cell['dataset'], references=compared))
    base.dump(output, dict(passed=True, rows=rows, source_sha256=base.digest(__file__)))


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['study', 'data-root', 'numpy-python']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--queue-pid', required=True, type=int)
    a = p.parse_args()
    study = Path(a.study)
    anchors_file = study/'anchors.json'
    anchors = json.loads(anchors_file.read_text())
    schedule = anchors.get('protocol') == 'schedule_order'
    count = 4 if schedule else 2
    assert len(anchors['references']) == 8 and len(anchors['new_cases']) == count
    evaluator = 'schedule_evaluate.py' if schedule else 'contrast_evaluate.py'
    assets = Path(anchors['assets'])
    assert base.digest(assets/'manifest.json') == anchors['assets_manifest_sha256']
    out = study/'analysis'
    out.mkdir(parents=True, exist_ok=False)
    def phase(name: str, **kw) -> None:
        base.dump(out/'status.json', dict(phase=name, pid=os.getpid(), unix_time=time.time(), **kw))
    try:
        terminal = study/'output/monitor/queue-finished.json'
        while not terminal.exists():
            os.kill(a.queue_pid, 0)
            phase('waiting_for_training', queue_pid=a.queue_pid)
            time.sleep(15)
        done = json.loads(terminal.read_text())
        assert not done['failed'] and not done['unstarted'] and len(done['completed']) == count
        assert all(r['exit_code'] == 0 and r['complete'] for r in done['completed'])
        for ref in anchors['references']:
            folder = Path(ref['folder'])
            for name, digest in ref['bound_files'].items():
                assert base.digest(folder/name) == digest
            for split in ['selection', 'confirmation']:
                original = Path(ref['evaluations'][split]['folder'])
                for name, digest in ref['evaluations'][split]['bound_files'].items():
                    assert base.digest(original/name) == digest
                target = out/split/ref['id']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(original, target_is_directory=True)
        pairing(anchors, out/'pairing.json')
        tasks = [(split, steps, row) for split, steps in
                 [('selection', ['15000', '20000']), ('confirmation', ['20000'])]
                 for row in anchors['new_cases']]
        def evaluate(task: tuple) -> dict:
            split, steps, row = task
            cmd = [sys.executable, str(HERE/evaluator), '--folder', row['folder'],
                '--data-root', a.data_root, '--assets', str(assets), '--output', str(out/split/row['id']),
                '--device', 'cpu', '--split', split, '--steps', *steps]
            with (out/(split+'-'+row['id']+'.log')).open('x') as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            complete = out/split/row['id']/'COMPLETE.json'
            assert proc.returncode == 0 and complete.exists(), (row['id'], split, proc.returncode)
            assert json.loads(complete.read_text())['passed']
            return dict(id=row['id'], split=split, exit_code=0,
                completion_sha256=base.digest(complete), command=cmd)
        receipts = []
        phase('evaluating', completed=0, total=2*count)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                receipts.append(future.result())
                base.dump(out/'receipts.json', receipts)
                phase('evaluating', completed=len(receipts), total=2*count)
        phase('independent_recomputation')
        code = ('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from analyze_screen import collect;'
                'r=Path(sys.argv[2]);n=int(sys.argv[3]);collect(r,r,split="selection",expected_cases=n);'
                'collect(r,r,split="confirmation",expected_cases=n)')
        subprocess.run([a.numpy_python, '-c', code, str(HERE.parent/'response_screen_20261115'),
                        str(out), str(8+count)], check=True)
        base.dump(out/'COMPLETE.json', dict(passed=True, training_cases=count, reused_reference_cases=8,
            selection_cases=8+count, confirmation_cases=8+count, anchors_sha256=base.digest(anchors_file),
            source_sha256=base.digest(__file__)))
        phase('complete', completed=2*count, total=2*count)
    except Exception as exc:
        phase('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
