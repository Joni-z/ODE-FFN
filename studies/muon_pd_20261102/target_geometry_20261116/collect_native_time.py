"""Wait for the exact native queue, then independently verify every case once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from geometry import read, dump, sha


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['input', 'study', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--wait', action='store_true')
    p.add_argument('--refined', action='store_true')
    a = p.parse_args()
    root, study, out = map(Path, [a.input, a.study, a.output])
    out.mkdir(parents=True, exist_ok=False)
    queue = read(root/'queue-start.json')
    terminal = root/'queue-finished.json'
    while not terminal.exists():
        assert a.wait, 'Native queue is not terminal'
        try:
            os.kill(queue['pid'], 0)
        except ProcessLookupError:
            if terminal.exists():
                break
            raise RuntimeError('Native queue process disappeared without a terminal receipt')
        command = subprocess.check_output(['ps', '-p', str(queue['pid']), '-o', 'command='], text=True)
        assert 'native_time_queue.py' in command and str(root) in command
        status = read(root/'status.json')
        dump(out/'progress.json', dict(phase='waiting_for_native_queue', collector_pid=os.getpid(),
            queue_pid=queue['pid'], completed=len(status['completed']), total=12, unix_time=time.time()))
        time.sleep(5)
    done = read(terminal)
    assert done['passed'] and len(done['completed']) == 12 and not done['unstarted']
    expected = {r['id'] for r in read(study/'anchors.json')['references']+read(study/'anchors.json')['new_cases']}
    assert {r['id'] for r in done['completed']} == expected
    assert all(r['exit_code'] == 0 and r['complete'] for r in done['completed'])
    summaries, receipts, bindings = [], [], []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for row in done['completed']:
        source, target = Path(row['output']), out/'cases'/row['id']
        native = read(source/'COMPLETE.json')
        assert not native['witness'] and native['sources'] == 256
        assert native['anchors_sha256'] == sha(study/'anchors.json')
        verifier = 'verify_refined_trajectory.py' if a.refined else 'verify_native_time.py'
        command = [sys.executable, str(Path(__file__).with_name(verifier)),
                   '--input', str(source), '--output', str(target)]
        with (out/(row['id']+'.log')).open('x') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        assert proc.returncode == 0, (row['id'], proc.returncode)
        proof = read(target/'COMPLETE.json')
        result = read(target/'summary.json')
        assert proof['passed'] and not proof['witness'] and result['all_source_arrays_exact']
        assert sha(target/'summary.json') == proof['summary_sha256']
        assert result['input_complete_sha256'] == sha(source/'COMPLETE.json')
        summaries.append(result)
        bindings.append(dict(id=row['id'], native_folder=str(source), native_complete_sha256=sha(source/'COMPLETE.json'),
            native_rows_sha256=sha(source/'rows.json'), verified_complete_sha256=sha(target/'COMPLETE.json'),
            verified_summary_sha256=sha(target/'summary.json')))
        receipts.append(dict(id=row['id'], command=command, actual_exit_code=proc.returncode))
        dump(out/'progress.json', dict(phase='verifying', collector_pid=os.getpid(), completed=len(summaries), total=12))
        print(f'{len(summaries)}/12 independently verified', flush=True)
    dump(out/'summary.json', dict(passed=True, rows=summaries, bindings=bindings,
        qualified_cases=sum(r['qualified'] for r in summaries),
        maximum_reconstruction_error=max(r['max_reconstruction_error'] for r in summaries)))
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, receipts=receipts,
        source_sha256=sha(Path(__file__)), queue_terminal_sha256=sha(terminal),
        summary_sha256=sha(out/'summary.json'), all_sources_retained=True))


if __name__ == '__main__':
    main()
