"""Publishable timestamped status from real process identities and saved records."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import psutil

def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default
def lines(path):
    if not path.exists():return []
    raw=path.read_text().splitlines();out=[]
    for i,line in enumerate(raw):
        try:out.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(raw)-1:raise
    return out
def alive(pid,created):
    try:return abs(psutil.Process(pid).create_time()-created)<.001 and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
def main(root):
    doc=Path(__file__).resolve().parent;now=datetime.now(timezone.utc).isoformat();phase=read(root/'orchestrator/status.json',{})
    identity=read(root/'orchestrator/identity.json',{});resources=lines(root/'output/resources.jsonl');latest=resources[-1]['resources'] if resources else []
    launches=read(root/'output/launches.json',[]);rows=[]
    for launch in launches:
        arm=launch['arm'];folder=root/'output/runs'/arm;status=read(folder/'status.json',{});metrics=lines(folder/'metrics.jsonl');streams=lines(folder/'streams.jsonl')
        usage=next((r for r in latest if r['pid']==launch['pid']),{})
        rows.append(dict(arm=arm,pid=launch['pid'],create_time=launch['create_time'],identity_live=alive(launch['pid'],launch['create_time']),
            recorded_state=status.get('state'),completed_updates=status.get('step',0),total_updates=20000,
            primed_CPU_percent=usage.get('CPU_percent'),RSS_bytes=usage.get('RSS'),
            latest_saved_validation=metrics[-1] if metrics else None,native_stream_checks=len(streams),exit_code=launch.get('exit_code')))
    value=dict(observed_at_UTC=now,phase=phase,controller_identity=identity,
        controller_identity_live=alive(identity['pid'],identity['create_time']) if identity else False,
        worker_rows=rows,quality_status='Pending complete pipeline, tool exit and final artifact audit',
        source_commit=read(root/'settings.json',{}).get('source_commit'))
    (doc/'progress.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    text=['# Recorded training progress','',f'Observed at `{now}`. This is a saved snapshot, not a live GitHub feed.','',
        f"Pipeline phase: `{phase.get('phase',phase.get('state','not started'))}`. Original controller identity live: `{value['controller_identity_live']}`.",'',
        '| Control | Completed updates | Process identity live | CPU (%) | RSS (GiB) | Latest saved validation step |',
        '|---|---:|---|---:|---:|---:|']
    for r in rows:
        cpu='NA' if r['primed_CPU_percent'] is None else format(r['primed_CPU_percent'],'.1f')
        memory='NA' if r['RSS_bytes'] is None else format(r['RSS_bytes']/2**30,'.3f');metric=r['latest_saved_validation']
        text.append(f"| {r['arm']} | {r['completed_updates']}/20000 | {r['identity_live']} | {cpu} | {memory} | {metric['step'] if metric else 'NA'} |")
    text+=['','All saved input-stream checks must match the sealed native trajectory. Training loss or validation regression alone does not establish a generative improvement.',
        '','The controller continues through boundary audit, seven-bank generation, independent sample replay, features and independent FID audit. Final conclusions require all terminal exits and artifact verification.',
        '',f'Private runtime: `{root}`. Source commit: `{value["source_commit"]}`.','']
    (doc/'progress.md').write_text('\n'.join(text));print(json.dumps(dict(observed_at_UTC=now,phase=phase.get('phase'),workers=rows)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
