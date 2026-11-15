"""Real kernel, original-parameter gradient and native resume witnesses."""
import argparse,importlib.util,json,subprocess,sys,time
from pathlib import Path
from coefficient_trial import HERE,TAG,base,torch,build_model,fixed_builder
import coefficient_evaluate as evaluation

spec=importlib.util.spec_from_file_location('mechanism_check_helpers',HERE.parent/'followup_20261115/check.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);base.verify_sources();rows=[]
    for dim in (2,16):
        torch.manual_seed(0);fixed=fixed_builder(dim,'screen_isru_k8').double();rng=torch.get_rng_state()
        torch.manual_seed(0);free=build_model(dim,'free_coeff_k8').double()
        assert torch.equal(rng,torch.get_rng_state())
        fp=dict(fixed.named_parameters());up=dict(free.named_parameters())
        assert free.parameter_count()-fixed.parameter_count()==42
        for n,v in fp.items():assert torch.equal(v,up[n]),n
        helpers.same(dict(fixed.named_buffers()),dict(free.named_buffers()))
        with torch.no_grad():
            w=torch.linspace(-.03,.04,fixed.output.weight.numel(),dtype=torch.double).reshape_as(fixed.output.weight)
            fixed.output.weight.copy_(w);free.output.weight.copy_(w)
        x=torch.randn(8,dim,dtype=torch.double,requires_grad=True);t=x.new_full((8,1),.3)
        left=fixed(x,t);right=free(x,t);assert torch.equal(left,right)
        gx=torch.autograd.grad(left.square().sum(),x,retain_graph=True)[0]
        gy=torch.autograd.grad(right.square().sum(),x,retain_graph=True)[0]
        assert torch.equal(gx,gy)
        left.square().mean().backward();right.square().mean().backward()
        for n,v in fp.items():
            assert (v.grad is None)==(up[n].grad is None),n
            if v.grad is not None:assert torch.equal(v.grad,up[n].grad),n
        for _,layer in free.ode_layers():
            assert bool(torch.isfinite(layer.alpha.grad).all()) and layer.alpha.grad.norm()>0
        layer=next(free.ode_layers())[1]
        h=torch.linspace(-1.1,1.3,64,dtype=torch.double,requires_grad=True)
        assert torch.autograd.gradcheck(lambda v:layer(v[None])[0],(h,),fast_mode=True)
        with torch.no_grad():layer.alpha.copy_(torch.linspace(.7,1.3,7,dtype=torch.double))
        matrix=layer.operator()*layer.step_size(h)[None,:];term=h;expected=h
        for k in range(1,9):
            term=matrix@term/k;expected=expected+(term if k==1 else layer.alpha[k-2]*term)
        assert torch.allclose(layer(h[None])[0],expected,atol=1e-12,rtol=1e-12)
        rows.append(dict(dim=dim,extra_scalars=42,initial_output_exact=True,original_parameter_gradients_exact=True,
                         input_gradients_exact=True,coefficient_gradients_nonzero=True,polynomial_and_gradcheck=True))
    base.dump(out/'kernels.json',dict(passed=True,rows=rows))
    for cell in json.loads((HERE/'grid.json').read_text())['cells']:
        def folder(root):return root/TAG/cell['dataset']/cell['id']/'seed000'
        def run(root,steps,label,resume=None):
            cmd=[sys.executable,str(HERE/'coefficient_trial.py'),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device','cpu','--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (out/(cell['id']+label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        run(out/'resume',8,'-8')
        run(out/'resume',16,'-resume16',folder(out/'resume')/'checkpoints/step_0000008.pt')
        run(out/'continuous',16,'-continuous16')
        left=torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt',weights_only=False)
        right=torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt',weights_only=False)
        helpers.numerical(left,right)
        cfg=json.loads((folder(out/'resume')/'config.json').read_text());cache=base.verify_data(a.data_root,cell)
        model,_=evaluation.evaluator.ev.restore(folder(out/'resume'),cfg,cache/'manifest.json',16,'cpu')
        with torch.no_grad():
            for _,layer in model.ode_layers():assert not torch.equal(layer.alpha,torch.ones_like(layer.alpha))
        cmd=[sys.executable,str(HERE/'coefficient_evaluate.py'),'--folder',str(folder(out/'resume')),'--data-root',a.data_root,'--assets',a.assets,'--output',str(out/('evaluation-'+cell['dataset'])),'--device','cpu','--steps','16','--witness']
        with (out/(cell['id']+'-evaluation.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    base.dump(out/'PASS.json',dict(passed=True,kernels=rows,native_continuations=2,real_evaluator_witnesses=2,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
