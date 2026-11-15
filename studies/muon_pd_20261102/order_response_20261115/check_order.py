"""Explicit PD Taylor map/Jacobian, parameter identity and native continuation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from order_trial import HERE,TAG,base,torch,build_model,k1_builder
from check import same,numerical
import evaluate_order as evaluator


def explicit(layer,x):
    d=layer.step_size(x[None])[0];a=layer.operator()*d[None,:]
    power=torch.eye(len(x),device=x.device,dtype=x.dtype);result=x
    for k in range(1,layer.orders+1):
        power=power@a/k;result=result+power@x
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);base.verify_sources();rows=[]
    for dim in (2,16):
        for family in ('pd_isru_k8','pd_tanh_k8'):
            torch.manual_seed(0);ref=k1_builder(dim,family[:-1]+'1');rng=torch.get_rng_state()
            torch.manual_seed(0);model=build_model(dim,family)
            same(rng,torch.get_rng_state());same(ref.state_dict(),model.state_dict());same(dict(ref.named_buffers()),dict(model.named_buffers()))
            assert model.parameter_count()==(56706 if dim==2 else 58512)
            model=model.to(a.device).double();layer=next(model.ode_layers())[1]
            for scale in (0.,.1,2.):
                h=torch.linspace(-1,1,64,device=a.device,dtype=torch.double)*scale
                y=layer(h[None])[0];expected=explicit(layer,h)
                assert torch.allclose(y,expected,rtol=1e-11,atol=1e-11)
                if dim==2:
                    j=torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h)
                    je=torch.autograd.functional.jacobian(lambda x:explicit(layer,x),h)
                    assert torch.allclose(j,je,rtol=1e-10,atol=1e-10)
            with torch.no_grad():model.output.weight.copy_(torch.linspace(-.03,.04,model.output.weight.numel(),device=a.device,dtype=torch.double).reshape_as(model.output.weight))
            x=torch.randn(8,dim,device=a.device,dtype=torch.double);t=x.new_full((8,1),.3)
            model(x,t).square().mean().backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0 for _,p in base.projection_parameters(model))
            state=dict(model={k:v.clone() for k,v in model.state_dict().items()},runtime_buffers={k:v.clone() for k,v in model.named_buffers()})
            torch.manual_seed(27);copy=build_model(dim,family).to(a.device).double().eval();base.train.restore_model(copy,state)
            assert torch.equal(copy(x,t),model(x,t))
            assert bool(torch.isfinite(base.train.generated_samples(copy,x,128)).all())
            rows.append(dict(dim=dim,family=family,parameters=model.parameter_count(),initialization_rng_exact=True,explicit_polynomial=True,full_diagonal_jacobian=dim==2,nonzero_gradients=True,restoration_exact=True))
    base.dump(out/'kernel.json',dict(passed=True,rows=rows,torch=torch.__version__,source_sha256=base.digest(__file__)))
    anchors=[]
    import numpy as np
    for anchor in json.loads((HERE/'anchors.json').read_text()):
        folder=Path(anchor['folder']);cfg=json.loads((folder/'config.json').read_text())
        assert base.digest(folder/'config.json')==anchor['config_sha256']
        assert cfg['steps']==20000 and cfg['batch_size']==512 and cfg['seeds']==[0]
        assert cfg['cell']['optimizer']=='adamw' and cfg['cell']['projection_lr']==.0003
        for step,digest in anchor['checkpoint_sha256'].items():
            assert base.digest(folder/'checkpoints'/f'step_{int(step):07d}.pt')==digest
        cache=base.verify_data(a.data_root,cfg['cell']);manifest=cache/'manifest.json'
        spec=json.loads(manifest.read_text())['datasets'][anchor['dataset']]
        law=base.train.load_distribution(spec)
        fixed=base.train.fixed_eval_data(cache/anchor['dataset'],law,law.dim,0,torch.device(a.device),4096,256,'test')
        model,identity=evaluator.parent.restore(folder,cfg,manifest,20000,a.device)
        generated=base.train.generated_samples(model,fixed['sample_source'],32).cpu().numpy()
        with np.load(folder/'final_test_examples.npz') as z:assert np.array_equal(generated,z['generated'])
        anchors.append(dict(family=anchor['family'],dataset=anchor['dataset'],native_sampling_exact=True,**identity))
    base.dump(out/'anchors.json',dict(passed=True,rows=anchors))
    grid=json.loads((HERE/'grid.json').read_text());done=[]
    def run(cell,root,steps,label,resume=None):
        cmd=[sys.executable,str(HERE/'order_trial.py'),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device',a.device,'--steps',str(steps)]
        if resume:cmd+=['--resume',str(resume)]
        with (out/(cell['id']+'-'+label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    def ckpt(root,c,step):return root/TAG/c['dataset']/c['id']/'seed000/checkpoints'/f'step_{step:07d}.pt'
    for c in grid['cells']:
        base.verify_data(a.data_root,c)
        run(c,out/'resume',8,'8')
        run(c,out/'resume',16,'resume16',ckpt(out/'resume',c,8))
        run(c,out/'continuous',16,'continuous16')
        s=torch.load(ckpt(out/'resume',c,16),map_location='cpu',weights_only=False)
        other=torch.load(ckpt(out/'continuous',c,16),map_location='cpu',weights_only=False);numerical(s,other)
        folder=ckpt(out/'resume',c,16).parents[1];cfg=json.loads((folder/'config.json').read_text())
        manifest=Path(cfg['data_root'])/'manifest.json'
        restored,identity=evaluator.parent.restore(folder,cfg,manifest,16,a.device)
        torch.manual_seed(0);original=build_model(2 if c['data_version']=='v1' else 16,c['family']).to(a.device).eval();base.train.restore_model(original,s)
        x=torch.randn(64,original.dim,device=a.device);t=x.new_full((len(x),1),.4)
        assert torch.equal(original(x,t),restored(x,t))
        assert torch.equal(base.train.generated_samples(original,x,32),base.train.generated_samples(restored,x,32))
        done.append(dict(cell=c['id'],resume16_exact=True,evaluator_roundtrip_exact=True,**identity))
        base.dump(out/'progress.json',dict(completed=done))
    base.dump(out/'PASS.json',dict(passed=True,completed=done));print(json.dumps(dict(passed=True,cells=len(done),output=str(out))))

if __name__=='__main__':main()
