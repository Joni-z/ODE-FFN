"""Toy derivatives and old positive-panel extraction before trained gradients."""
import argparse,time
import numpy as np
import torch
from study_support import build,layers,native_check,write,sha,MODELS,DATASET
from tangent_math import self_test

def main(root):
    torch.set_num_threads(1);start=time.monotonic();model,cp,_=build();digest=sha(cp);native=native_check(model,cp);model=model.double();state={};hooks=[]
    for name,layer in layers(model):
        def logit_hook(m,a,y,key=name,l=layer):state.setdefault(key,{})['duration']=torch.sigmoid(y/l.tau)*l.scale+l.shift
        def layer_hook(m,a,y,key=name):state.setdefault(key,{}).update(x=a[0],y=y)
        hooks.append(layer.t_func[-1].register_forward_hook(logit_hook));hooks.append(layer.register_forward_hook(layer_hook))
    with np.load(MODELS/'banks'/DATASET/'panel_0.5.npz') as bank:
        x=torch.from_numpy(bank['x'][:256].astype(np.float64));t=torch.full((len(x),1),.5,dtype=torch.float64)
    with torch.no_grad():
        value=model(x,t)
        for name,layer in layers(model):
            s=state[name];p=layer.proj.weight;one=torch.nn.functional.linear(s['x'],p)*s['duration']
            two=torch.nn.functional.linear(one,p)*s['duration']/2
            torch.testing.assert_close(s['y'],(s['x']+one)+two,atol=0,rtol=0)
    for hook in hooks:hook.remove()
    with torch.no_grad():torch.testing.assert_close(value,model(x,t),atol=0,rtol=0)
    assert sha(cp)==digest
    result=dict(status='PASS',elapsed_seconds=time.monotonic()-start,native_float32_panels=native,positive_hook_cases=6,
        checkpoint_sha256=digest,self_tests=self_test(),new_trained_P_route_measurements=0,new_model_updates=0)
    assert not (root/'preflight.json').exists();write(root/'preflight.json',result);print(result,flush=True)

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
