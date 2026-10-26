"""Attach read-only 10s telemetry to an already-running local study queue."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import psutil


def monitor(root,queue_pid):
    current=psutil.Process(queue_pid)
    expected_creation=current.create_time()
    assert 'run_architecture_study.py' in ' '.join(current.cmdline())
    while True:
        try:queue=json.loads((root/'queue/status.json').read_text())
        except json.JSONDecodeError:
            time.sleep(1);continue
        processes=[]
        try:
            assert current.create_time()==expected_creation
            for child in [current,*current.children(recursive=True)]:
                try:
                    processes.append(dict(pid=child.pid,command=child.cmdline(),
                        rss_mib=child.memory_info().rss/1024**2,cpu_times=child.cpu_times()._asdict()))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        except psutil.NoSuchProcess:pass
        studies=[]
        for path in sorted(root.glob('*/status.json')):
            if path.parent.name=='queue':continue
            try:studies.append(dict(study=path.parent.name,**json.loads(path.read_text())))
            except json.JSONDecodeError:continue
        row=dict(time=datetime.now(timezone.utc).isoformat(),queue=queue,studies=studies,
            cpu_percent=psutil.cpu_percent(None),memory_available_gib=psutil.virtual_memory().available/1024**3,
            processes=processes,psutil=psutil.__version__,device='cpu',
            scope='Attached after the initial v1 monitoring thread failed for a missing dependency; no retrospective telemetry is inferred.')
        destination=root/'resource-monitor.json'
        temp=root/'resource-monitor.json.tmp';temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(destination)
        with (root/'resource-telemetry.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        if queue['state'] in ('COMPLETE','FAILED'):return
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--queue-pid',type=int,required=True)
    args=parser.parse_args();monitor(args.output_root,args.queue_pid)
