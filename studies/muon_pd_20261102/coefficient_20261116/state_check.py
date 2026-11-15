"""Check midpoint algebra, full gradients, common tensors and native continuation."""
import argparse,importlib.util,json,subprocess,sys
from state_trial import HERE,TAG,base,torch,build_model,fixed_builder
import coefficient_evaluate as evaluation
spec=importlib.util.spec_from_file_location('state_check_helpers',HERE.parent/'followup_20261115/check.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=__import__('pathlib').Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);rows=[]
    for dim in (2,16):
        torch.manual_seed(0);ref=fixed_builder(dim,'screen_isru_k8');rng=torch.get_rng_state()
        torch.manual_seed(0);model=build_model(dim,'state_midpoint4')
        helpers.same(ref.state_dict(),model.state_dict());helpers.same(dict(ref.named_buffers()),dict(model.named_buffers()))
        assert torch.equal(rng,torch.get_rng_state()) and model.parameter_count()==ref.parameter_count()
        model=model.double();layer=next(model.ode_layers())[1];h=torch.linspace(-1,1.1,64,dtype=torch.double,requires_grad=True)
        y=h
        f=lambda x:layer.operator()@(x/torch.sqrt(1+1.2*x*x))
        for _ in range(4):y=y+.25*f(y+.125*f(y))
        assert torch.allclose(layer(h[None])[0],y,atol=1e-12,rtol=1e-12)
        assert torch.autograd.gradcheck(lambda x:layer(x[None])[0],(h,),fast_mode=True)
        reference=layer.solve(h[None],128)
        errors=[float((layer.solve(h[None],n)-reference).norm().detach()) for n in [4,8,16]]
        assert errors[0]>3*errors[1] and errors[1]>3*errors[2]
        rows.append(dict(dim=dim,common_tensors_rng_exact=True,parameters_equal=True,midpoint_formula=True,full_gradient=True,refinement_errors=errors))
    for c in json.loads((HERE/'state-grid.json').read_text())['cells']:
        def folder(root):return root/TAG/c['dataset']/c['id']/'seed000'
        def run(root,steps,label,resume=None):
            cmd=[sys.executable,str(HERE/'state_trial.py'),'--cell',c['id'],'--data-root',a.data_root,'--output',str(root),'--device','cpu','--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (out/(c['id']+label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        run(out/'resume',8,'-8');run(out/'resume',16,'-resume16',folder(out/'resume')/'checkpoints/step_0000008.pt');run(out/'continuous',16,'-continuous16')
        l=torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt',weights_only=False);r=torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt',weights_only=False);helpers.numerical(l,r)
        cmd=[sys.executable,str(HERE/'state_evaluate.py'),'--folder',str(folder(out/'resume')),'--data-root',a.data_root,'--assets',a.assets,'--output',str(out/('evaluation-'+c['dataset'])),'--device','cpu','--steps','16','--witness']
        with (out/(c['id']+'-eval.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    base.dump(out/'PASS.json',dict(passed=True,kernels=rows,native_continuations=2,real_evaluator_witnesses=2,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
