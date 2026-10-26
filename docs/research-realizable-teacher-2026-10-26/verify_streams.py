"""Delta check for actual consumed cache hashes, without repeating native tests."""
import argparse,hashlib
from pathlib import Path
import torch
from teacher_support import STUDENTS,TEACHERS,write,student,optimizer,open_cache,batch,update,snapshot,restore,assert_equal,replay_hash
from prepare_cache import prepare

def main(root):
    torch.set_num_threads(1);folder=root/'stream-preflight';folder.mkdir(exist_ok=False);meta=prepare(folder/'cache',steps=3);cache=open_cache(folder/'cache');checks=[]
    for teacher_name in TEACHERS:
        initial=hashlib.sha256(b'').hexdigest();actual=dict(input=initial,label=initial)
        for step in range(3):
            x,t,y=batch(cache,teacher_name,step);actual['input']=replay_hash(actual['input'],x,t);actual['label']=replay_hash(actual['label'],y)
            if step+1 in [2,3]:
                expected=meta['snapshots'][str(step+1)];assert actual==dict(input=expected['input_replay_digest'],label=expected['label_replay_digests'][teacher_name])
        checks.append(dict(teacher=teacher_name,three_actual_input_label_steps='PASS'))
    m=student('ode_k2');opt=optimizer(m);actual=dict(input=initial,label=initial)
    for i in range(2):
        x,t,y=batch(cache,'ode_k2',i);update(m,opt,x,t,y);actual['input']=replay_hash(actual['input'],x,t);actual['label']=replay_hash(actual['label'],y)
    run=folder/'snapshot';run.mkdir();snapshot(m,opt,2,run,meta,'ode_k2',actual.copy())
    state=torch.load(run/'step_0000002.pt',map_location='cpu',weights_only=False);assert state['actual_replay_digests']==actual
    resumed=student('ode_k2');ropt=optimizer(resumed);restore(resumed,ropt,state)
    x,t,y=batch(cache,'ode_k2',2);update(m,opt,x,t,y);update(resumed,ropt,x,t,y);assert_equal(m.state_dict(),resumed.state_dict());assert_equal(opt.state_dict(),ropt.state_dict())
    result=dict(status='PASS',checks=checks,replay_hash_checkpoint_resume='bitwise',first_preflight_preserved=True,new_formal_student_updates=0)
    write(root/'preflight-streams.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
