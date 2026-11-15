"""Single-generator release: initialization, gradients and native continuation."""
import argparse,importlib.util,json,subprocess,sys
from pathlib import Path
from sharing_trial import HERE,TAG,base,torch,build_model,state_builder
spec=importlib.util.spec_from_file_location('sharing_check_helpers',HERE.parent/'followup_20261115/check.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False);torch.set_num_threads(1);rows=[]
    for dim in [2,16]:
        torch.manual_seed(0);fixed=state_builder(dim,'state_midpoint4').double();rng=torch.get_rng_state()
        torch.manual_seed(0);free=build_model(dim,'stage_operator_midpoint4').double();assert torch.equal(rng,torch.get_rng_state())
        fp=dict(fixed.named_parameters());up=dict(free.named_parameters());assert free.parameter_count()-fixed.parameter_count()==73728
        for n,v in fp.items():assert torch.equal(v,up[n]),n
        helpers.same(dict(fixed.named_buffers()),dict(free.named_buffers()))
        with torch.no_grad():
            w=torch.linspace(-.03,.04,fixed.output.weight.numel(),dtype=torch.double).reshape_as(fixed.output.weight)
            fixed.output.weight.copy_(w);free.output.weight.copy_(w)
        x=torch.randn(8,dim,dtype=torch.double,requires_grad=True);t=x.new_full((len(x),1),.3)
        left=fixed(x,t);right=free(x,t);assert torch.equal(left,right)
        gx=torch.autograd.grad(left.square().sum(),x,retain_graph=True)[0];gy=torch.autograd.grad(right.square().sum(),x,retain_graph=True)[0]
        assert torch.allclose(gx,gy,atol=1e-14,rtol=1e-10)
        left.square().mean().backward();right.square().mean().backward();maximum=0.
        for n,v in fp.items():
            assert (v.grad is None)==(up[n].grad is None),n
            if v.grad is not None:
                maximum=max(maximum,float((v.grad-up[n].grad).abs().max()))
                assert torch.allclose(v.grad,up[n].grad,atol=1e-14,rtol=1e-10),n
        for _,layer in free.ode_layers():assert torch.isfinite(layer.stage_delta.grad).all() and layer.stage_delta.grad.norm()>0
        layer=next(free.ode_layers())[1];h=torch.linspace(-1.1,1.3,64,dtype=torch.double,requires_grad=True)
        assert torch.autograd.gradcheck(lambda v:layer(v[None])[0],(h,),fast_mode=True)
        with torch.no_grad():
            layer.stage_delta.copy_(torch.linspace(-.02,.03,layer.stage_delta.numel(),dtype=torch.double).reshape_as(layer.stage_delta))
        y=h
        for i in range(4):
            weight=layer.operator() if i==0 else layer.operator()+layer.stage_delta[i-1]
            f=lambda v:weight@(v/torch.sqrt(1+1.2*v*v))
            y=y+.25*f(y+.125*f(y))
        assert torch.allclose(layer(h[None])[0],y,atol=1e-12,rtol=1e-12)
        rows.append(dict(dim=dim,extra_parameters=73728,initial_output_exact=True,common_gradient_max_error=maximum,new_gradients_nonzero=True,formula_and_gradcheck=True))
    for c in json.loads((HERE/'sharing-grid.json').read_text())['cells']:
        def folder(root):return root/TAG/c['dataset']/c['id']/'seed000'
        def run(root,steps,label,resume=None):
            cmd=[sys.executable,str(HERE/'sharing_trial.py'),'--cell',c['id'],'--data-root',a.data_root,'--output',str(root),'--device','cpu','--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (out/(c['id']+label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        run(out/'resume',8,'-8');run(out/'resume',16,'-resume16',folder(out/'resume')/'checkpoints/step_0000008.pt');run(out/'continuous',16,'-continuous16')
        l=torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt',weights_only=False);r=torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt',weights_only=False);helpers.numerical(l,r)
        cmd=[sys.executable,str(HERE/'sharing_evaluate.py'),'--folder',str(folder(out/'resume')),'--data-root',a.data_root,'--assets',a.assets,'--output',str(out/('evaluation-'+c['dataset'])),'--device','cpu','--steps','16','--witness']
        with (out/(c['id']+'-eval.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    base.dump(out/'PASS.json',dict(passed=True,kernels=rows,native_continuations=2,real_evaluator_witnesses=2,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
