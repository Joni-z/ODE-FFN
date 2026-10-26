"""Exact teacher representers, cold starts, cached/live training and resume."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from teacher_support import TEACHERS,STUDENTS,BANKS,load,write,student,teacher,signature,verify_cold,optimizer,update,assert_equal,open_cache,batch,snapshot,restore,predict
from prepare_cache import prepare
from explicit_forward import forward

def main(root):
    torch.set_num_threads(1);started=time.monotonic();smoke=root/'preflight';smoke.mkdir(exist_ok=False)
    meta=prepare(smoke/'cache',steps=3);cache=open_cache(smoke/'cache');checks=[];costs={}
    from synthetic.conservative_readout import compute_budget
    with np.load(BANKS/'field.npz') as b:probe_x=b['x'][:128];probe_t=b['t'][:128]
    for target_name in TEACHERS:
        teacher_model,teacher_cp=teacher(target_name);teacher_before=signature(teacher_model)
        exact=student(target_name)
        from synthetic.train import restore_model
        restore_model(exact,torch.load(teacher_cp,map_location='cpu',weights_only=False));exact.eval()
        np.testing.assert_array_equal(predict(exact,probe_x,probe_t),predict(teacher_model,probe_x,probe_t))
        for family in STUDENTS:
            cold=student(family);verify_cold(cold,family);initial=copy.deepcopy(cold.state_dict());fullopt=optimizer(cold)
            costs[family]=compute_budget(cold)
            for i in range(3):update(cold,fullopt,*batch(cache,target_name,i))
            resumed=student(family);opt=optimizer(resumed)
            for i in range(2):update(resumed,opt,*batch(cache,target_name,i))
            folder=smoke/(target_name+'--'+family);folder.mkdir();snapshot(resumed,opt,2,folder,meta,target_name)
            saved=torch.load(folder/'step_0000002.pt',map_location='cpu',weights_only=False)
            resumed=student(family);opt=optimizer(resumed);restore(resumed,opt,saved);update(resumed,opt,*batch(cache,target_name,2))
            assert_equal(resumed.state_dict(),cold.state_dict());assert_equal(opt.state_dict(),fullopt.state_dict())
            live=student(family);liveopt=optimizer(live)
            for i in range(3):
                x,t,cache_y=batch(cache,target_name,i)
                with torch.no_grad():live_y=teacher_model(x,t)
                torch.testing.assert_close(live_y,cache_y,atol=0,rtol=0);update(live,liveopt,x,t,live_y)
            assert_equal(live.state_dict(),cold.state_dict());assert_equal(liveopt.state_dict(),fullopt.state_dict())
            xx=torch.from_numpy(probe_x.copy()).requires_grad_(True);tt=torch.from_numpy(probe_t.copy())
            output=cold(xx,tt);explicit=forward(cold,xx,tt,family);torch.testing.assert_close(output,explicit,atol=0,rtol=0)
            dx=torch.autograd.grad(output.sum(),xx,retain_graph=True)[0];dy=torch.autograd.grad(explicit.sum(),xx)[0]
            torch.testing.assert_close(dx,dy,atol=1e-6,rtol=1e-6)
            changed=[name for name,value in cold.state_dict().items() if not torch.equal(value,initial[name])]
            if family=='ode_k2':
                assert any('.proj.weight' in name for name in changed);assert any('.t_func.' in name for name in changed)
            else:assert any('.w12.' in name for name in changed)
            checks.append(dict(teacher=target_name,student=family,resume='bitwise',live_cached_training='bitwise',explicit_forward='bitwise',changed_tensors=len(changed)))
        assert signature(teacher_model)==teacher_before and all(p.grad is None for p in teacher_model.parameters())
    from run_study import observe
    heartbeat=observe();assert heartbeat['processes'] and heartbeat['available_memory_bytes']>0;write(smoke/'monitor-heartbeat.json',heartbeat)
    result=dict(status='PASS',elapsed_seconds=time.monotonic()-started,old_teacher_native_panels=meta['old_native_panels_replayed'],old_teacher_native_fields=meta['old_native_fields_replayed'],
        cold_student_checks=checks,exact_teacher_representers=2,costs=costs,actual_monitor_heartbeat=True,new_formal_student_updates=0)
    assert not (root/'preflight.json').exists();write(root/'preflight.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
