"""Read-only audit of every retained full-batch prefix before CPU handoff."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from verify_numerical import sha, read


def live(pid: int) -> bool:
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and state[0] not in 'ZT'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    study, output = Path(a.study), Path(a.output)
    assert not output.exists()
    old, new = study/'numerical-full', study/'numerical-bankwise'
    status_digest = sha(old/'status.json')
    old_status, new_status = read(old/'status.json'), read(new/'status.json')
    for launch_name, repo in [('numerical-launch.json', study/'observer-repo'),
                             ('bankwise-launch.json', study/'bankwise-repo')]:
        launch = read(study/launch_name)
        status = old_status if launch_name == 'numerical-launch.json' else new_status
        assert status['pid'] == launch['pid']
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == launch['frozen_commit']
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
    cases = {c['id']: c for c in read(study/'portable/sampling-models/portable.json')['cases']}
    compared, missing, bindings = [], [], []
    for task in old_status['completed']+old_status['running']:
        folder = Path(task['output'])
        file = folder/'COMPLETE.json' if (folder/'COMPLETE.json').exists() else folder/'progress.json'
        if not file.exists():
            continue
        binding = sha(file)
        recorded = read(file)['rows']
        newer = new/task['id']/task['mode']
        nf = newer/'COMPLETE.json' if (newer/'COMPLETE.json').exists() else newer/'progress.json'
        lookup = {(r['inner_substeps'], r['heun_steps']): r for r in read(nf)['rows']} if nf.exists() else {}
        for row in recorded:
            key = (row['inner_substeps'], row['heun_steps'])
            if key not in lookup:
                missing.append(dict(id=task['id'], mode=task['mode'], inner=key[0], outer=key[1]))
                continue
            candidate = lookup[key]
            case = cases[task['id']]
            for name in ['id', 'dataset', 'family', 'mode', 'inner_substeps', 'heun_steps', 'samples', 'checkpoint_sha256', 'assets_sha256']:
                assert row[name] == candidate[name], (task['id'], name)
            assert row['samples'] == 32768
            assert row['checkpoint_sha256'] == case['checkpoint_sha256'] and row['assets_sha256'] == case['assets_sha256']
            left, right = folder/row['generated_file'], newer/candidate['generated_file']
            assert sha(left) == row['generated_sha256'] and sha(right) == candidate['generated_sha256']
            x, y = np.load(left, allow_pickle=False), np.load(right, allow_pickle=False)
            assert x.dtype == y.dtype and np.array_equal(x, y), (task['id'], key)
            error = max(abs(row['pooled_sw2']-candidate['pooled_sw2']),
                        float(np.max(np.abs(np.asarray(row['bank_sw2'])-candidate['bank_sw2']))))
            assert error < 1e-12
            compared.append(dict(id=task['id'], mode=task['mode'], inner=key[0], outer=key[1],
                exact_samples=True, score_max_error=error, source_sample_sha256=sha(left)))
        bindings.append(dict(file=str(file), sha256=binding))
    stable = sha(old/'status.json') == status_digest and all(sha(Path(b['file'])) == b['sha256'] for b in bindings)
    advancing = []
    for task in new_status['running']:
        file = Path(task['output'])/'sampling-progress.json'
        if live(task['pid']) and file.exists() and read(file)['samples_completed'] > 0:
            advancing.append(task['pid'])
    terminal = new/'queue-finished.json'
    completed = terminal.exists() and read(terminal)['passed'] and not read(terminal)['unstarted']
    eligible = bool(compared) and not missing and stable and not new_status['failed'] and (bool(advancing) or completed)
    result = dict(passed=True, scope='retained-prefix comparison; not full sweep completion',
        unix_time=time.time(), old_queue_pid=old_status['pid'], old_queue_live=live(old_status['pid']),
        new_queue_pid=new_status['pid'], advancing_new_workers=advancing,
        new_queue_failed=new_status['failed'], old_status_sha256=status_digest,
        old_snapshot_stable=stable, compared_rows=len(compared), missing_rows=missing,
        eligible_to_retire_old=eligible, comparisons=compared, old_prefix_bindings=bindings,
        source_sha256=sha(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons', 'old_prefix_bindings']}))


if __name__ == '__main__':
    main()
