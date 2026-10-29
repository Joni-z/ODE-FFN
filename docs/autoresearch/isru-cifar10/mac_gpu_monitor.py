"""Read-only Apple GPU telemetry for this owned trainer; no training changes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import plistlib
import subprocess
import time
from pathlib import Path
import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    out = a.root/'output'
    record = json.loads((out/'process.json').read_text())
    process = psutil.Process(record['training_pid'])
    expected_path = str(a.root/'source/train.py')
    assert expected_path in process.cmdline()
    created = process.create_time()
    assert not (out/'apple-gpu-monitor.jsonl').exists(), 'preserve existing observations'
    with Path(__file__).open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = dict(pid=os.getpid(), target_pid=process.pid, target_created=created,
        source_sha256=source_hash, counter='ioreg AGXAccelerator PerformanceStatistics; whole device',
        interval_seconds=30, policy='read-only; stop at trainer exit; no restart or remote messages')
    (out/'apple-gpu-monitor-process.json').write_text(json.dumps(metadata, indent=2)+'\n')
    while process.is_running() and process.create_time() == created:
        row = dict(timestamp=time.time(), training_pid=process.pid)
        try:
            result = subprocess.run(['/usr/sbin/ioreg', '-a', '-r', '-c', 'AGXAccelerator'],
                                    capture_output=True, check=True, timeout=10)
            devices = plistlib.loads(result.stdout)
            row['devices'] = [{k: v for k, v in d.get('PerformanceStatistics', {}).items()
                               if 'Utilization' in k or 'memory' in k.lower()} for d in devices]
        except Exception as exc:
            row['unavailable'] = repr(exc)
        with (out/'apple-gpu-monitor.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        tmp = out/'apple-gpu-monitor-latest.json.tmp'
        tmp.write_text(json.dumps(row, indent=2)+'\n')
        os.replace(tmp, out/'apple-gpu-monitor-latest.json')
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except psutil.NoSuchProcess:
        pass
