"""Analytical derivative, unchanged initialization, native trainer/resume witness."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from train import HERE, TAG, base, build_model, previous_builder, torch


def same(a, b):
    if torch.is_tensor(a): assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else: assert a == b


def numerical(a, b):
    assert a['step'] == b['step']
    for key in ('model','runtime_buffers','optimizer','data_rng','torch_rng',
                'cuda_rng','numpy_rng','python_rng'): same(a[key], b[key])


def main():
    p=argparse.ArgumentParser();p.add_argument('--device', default='cuda')
    p.add_argument('--data-root', required=True);p.add_argument('--output', required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);base.verify_sources(); rows=[]
    for dim in (2,16):
        torch.manual_seed(0);ref=previous_builder(dim,'pd_isru_k1');rng=torch.get_rng_state()
        torch.manual_seed(0);model=build_model(dim,'pd_tanh_k1');same(rng,torch.get_rng_state())
        same(ref.state_dict(),model.state_dict());same(dict(ref.named_buffers()),dict(model.named_buffers()))
        model=model.to(args.device).double();layer=next(model.ode_layers())[1]
        for scale in (0.,1e-5,.1,1.,1000.):
            h=torch.linspace(-2,2,64,device=args.device,dtype=torch.double)*scale
            q=torch.tanh(1.2**.5*h)/1.2**.5
            expected=h+layer.operator()@q
            assert torch.allclose(layer(h[None])[0],expected,atol=1e-11,rtol=1e-11)
            jac=torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h)
            exact=torch.eye(64,device=args.device,dtype=torch.double)+layer.operator()@torch.diag(1-torch.tanh(1.2**.5*h).square())
            assert torch.allclose(jac,exact,atol=1e-10,rtol=1e-10)
        scalar=torch.zeros((),device=args.device,dtype=torch.double,requires_grad=True)
        q=scalar*layer.step_size(scalar)
        for _ in range(3):q=torch.autograd.grad(q,scalar,create_graph=True)[0]
        assert abs(float(q.detach())+2.4)<1e-12
        with torch.no_grad():model.output.weight.fill_(.01)
        x=torch.randn(8,dim,device=args.device,dtype=torch.double)
        model(x,torch.ones(8,1,device=args.device,dtype=torch.double)*.3).square().mean().backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0 for _,p in base.projection_parameters(model))
        rows.append(dict(dim=dim,parameters=model.parameter_count(),initialization_exact=True,full_jacobian=True,third_derivative=-2.4))
    base.dump(out/'kernel.json',dict(passed=True,rows=rows,torch=torch.__version__,device=args.device))
    grid=json.loads((HERE/'grid.json').read_text()); completed=[]
    def execute(runner,cell,root,steps,log,resume=None):
        cmd=[sys.executable,str(runner),'--cell',cell,'--data-root',args.data_root,'--output',str(root),'--device',args.device,'--steps',str(steps)]
        if resume:cmd+=['--resume',str(resume)]
        with (out/log).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    def state(root,tag,cell,step):
        return root/tag/cell['dataset']/cell['id']/'seed000/checkpoints'/f'step_{step:07d}.pt'
    def load(path):return torch.load(path,map_location='cpu',weights_only=False)
    for cell in grid['cells']:
        cid=cell['id'];base.verify_data(args.data_root,cell)
        execute(HERE/'train.py',cid,out/'resume',8,cid+'-8.log')
        execute(HERE/'train.py',cid,out/'resume',16,cid+'-resume16.log',state(out/'resume',TAG,cell,8))
        execute(HERE/'train.py',cid,out/'continuous',16,cid+'-continuous16.log')
        numerical(load(state(out/'resume',TAG,cell,16)),load(state(out/'continuous',TAG,cell,16)))
        if cell['role']=='same_recipe_replay':
            execute(HERE.parent/'autoresearch_20261103/cycle13/trial13.py',cell['source_id'],out/'parent',8,cid+'-parent8.log')
            old=dict(cell,id=cell['source_id'])
            numerical(load(state(out/'resume',TAG,cell,8)),load(state(out/'parent','professor_autoresearch_cycle13_20261103',old,8)))
        completed.append(dict(cell=cid,resume16_exact=True,parent8_exact=cell['role']=='same_recipe_replay'))
        base.dump(out/'progress.json',dict(completed=completed))
    base.dump(out/'PASS.json',dict(passed=True,completed=completed,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cells=len(completed),output=str(out))),flush=True)

if __name__=='__main__':main()
