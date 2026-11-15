"""Select an actually idle145GPU4-7 and bind one guarded inference process."""
import argparse,csv,io,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def eligible(rows,occupied):
    return [(int(r[0]),r[1].strip()) for r in rows if int(r[0]) in [4,5,6,7] and float(r[2])<500 and r[1].strip() not in occupied]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--witness',action='store_true')
    p.add_argument('--readiness');p.add_argument('--include-inner16',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    assert not (a.witness and a.detach)
    if not a.witness:
        assert a.readiness and json.loads(Path(a.readiness).read_text())['passed']
    out=Path(a.output);record=out.with_name(out.name+'-launch.json');assert not out.exists() and not record.exists()
    query=lambda fields:subprocess.check_output(['nvidia-smi','--query-'+fields,'--format=csv,noheader,nounits'],text=True)
    rows=list(csv.reader(io.StringIO(query('gpu=index,uuid,memory.used'))))
    occupied={r[0].strip() for r in csv.reader(io.StringIO(query('compute-apps=gpu_uuid,pid'))) if len(r)>1 and r[1].strip().isdigit()}
    idle=eligible(rows,occupied)
    if not idle:print(json.dumps(dict(started=False,reason='No idle physical GPU4-7')));raise SystemExit(75)
    index,uuid=idle[0];cmd=[sys.executable,'-u',str(HERE/('gpu_resolution.py' if a.witness else 'sharing_gpu.py')),'--root',a.root,'--output',a.output,'--gpu-uuid',uuid]
    if a.witness:cmd+=['--witness']
    else:
        cmd+=['--readiness',a.readiness]
        if a.include_inner16:cmd+=['--include-inner16']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    console=out.with_name(out.name+'-console.log');head=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip()
    with console.open('x') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,start_new_session=a.detach)
        launch=dict(pid=proc.pid,physical_gpu=index,gpu_uuid=uuid,command=cmd,frozen_commit=head,unix_time=time.time(),witness=a.witness,console=str(console))
        record.write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
        if not a.detach:
            code=proc.wait();launch['exit_code']=code;record.write_text(json.dumps(launch,indent=2)+'\n');raise SystemExit(code)

if __name__=='__main__':main()
