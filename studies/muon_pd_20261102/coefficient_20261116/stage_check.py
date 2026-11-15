"""Matched-function/gradient check and native continuation/evaluation witnesses."""
import argparse,importlib.util,json,subprocess,sys
from pathlib import Path
from stage_trial import HERE,TAG,base,torch,build_model,state_builder
spec=importlib.util.spec_from_file_location('stage_check_helpers',HERE.parent/'followup_20261115/check.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(1);rows=[]
    for dim in [2,16]:
        torch.manual_seed(0);fixed=state_builder(dim,'state_midpoint4').double();rng=torch.get_rng_state()
        torch.manual_seed(0);free=build_model(dim,'free_stage_midpoint4').double();assert torch.equal(rng,torch.get_rng_state())
        fp=dict(fixed.named_parameters());up=dict(free.named_parameters());assert free.parameter_count()-fixed.parameter_count()==24
        for n,v in fp.items():assert torch.equal(v,up[n]),n
        helpers.same(dict(fixed.named_buffers()),dict(free.named_buffers()))
        with torch.no_grad():
            w=torch.linspace(-.03,.04,fixed.output.weight.numel(),dtype=torch.double).reshape_as(fixed.output.weight)
            fixed.output.weight.copy_(w);free.output.weight.copy_(w)
        x=torch.randn(8,dim,dtype=torch.double,requires_grad=True);t=x.new_full((len(x),1),.3)
        left=fixed(x,t);right=free(x,t);assert torch.equal(left,right)
        gx=torch.autograd.grad(left.square().sum(),x,retain_graph=True)[0];gy=torch.autograd.grad(right.square().sum(),x,retain_graph=True)[0];assert torch.equal(gx,gy)
        left.square().mean().backward();right.square().mean().backward()
        for n,v in fp.items():
            assert (v.grad is None)==(up[n].grad is None),n
            if v.grad is not None:assert torch.equal(v.grad,up[n].grad),n
        for _,layer in free.ode_layers():assert torch.isfinite(layer.stage_gamma.grad).all() and layer.stage_gamma.grad.norm()>0
        layer=next(free.ode_layers())[1];h=torch.linspace(-1.1,1.3,64,dtype=torch.double,requires_grad=True)
        assert torch.autograd.gradcheck(lambda v:layer(v[None])[0],(h,),fast_mode=True)
        with torch.no_grad():layer.stage_gamma.copy_(torch.tensor([.7,.8,1.1,1.3],dtype=torch.double))
        y=h
        f=lambda v:layer.operator()@(v/torch.sqrt(1+1.2*v*v))
        for g in layer.stage_gamma:y=y+.25*f(y+.125*g*f(y))
        assert torch.allclose(layer(h[None])[0],y,atol=1e-12,rtol=1e-12)
        rows.append(dict(dim=dim,extra_parameters=24,initial_function_and_common_gradients_exact=True,new_gradients_nonzero=True,formula_and_gradcheck=True))
    for c in json.loads((HERE/'stage-grid.json').read_text())['cells']:
        def folder(root):return root/TAG/c['dataset']/c['id']/'seed000'
        def run(root,steps,label,resume=None):
            cmd=[sys.executable,str(HERE/'stage_trial.py'),'--cell',c['id'],'--data-root',a.data_root,'--output',str(root),'--device','cpu','--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (out/(c['id']+label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        run(out/'resume',8,'-8');run(out/'resume',16,'-resume16',folder(out/'resume')/'checkpoints/step_0000008.pt');run(out/'continuous',16,'-continuous16')
        left=torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt',weights_only=False);right=torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt',weights_only=False);helpers.numerical(left,right)
        cmd=[sys.executable,str(HERE/'stage_evaluate.py'),'--folder',str(folder(out/'resume')),'--data-root',a.data_root,'--assets',a.assets,'--output',str(out/('evaluation-'+c['dataset'])),'--device','cpu','--steps','16','--witness']
        with (out/(c['id']+'-eval.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    base.dump(out/'PASS.json',dict(passed=True,kernels=rows,native_continuations=2,real_evaluator_witnesses=2,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
