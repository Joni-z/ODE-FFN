"""Rank-matched controller algebra and full-map zero-anchor two-jet witness."""
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
from trial11 import base,build_model,controller_builder,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle6'))
from prefix import same,numerical_equal

def verify(device):
    base.verify_sources();torch.set_num_threads(1)
    cohort=json.loads(Path(__file__).with_name('factorial-parents.json').read_text())
    for f,digest in cohort['sources'].items():assert base.digest(base.HERE.parents[1]/f)==digest
    assert len(cohort['rows'])==18
    parents=[]
    for number in [1,2,3]:parents+=list(csv.DictReader((base.HERE.parents[1]/f'docs/autoresearch/professor-20261103/cycle{number}-results/endpoints.csv').open()))
    grid=json.loads(Path(__file__).with_name('grid.json').read_text());assert len(grid['cells'])==10
    for cell in grid['cells']:
        parent=next(r for r in parents if r['id']==cell['parent_id'] and r['checkpoint_sha256']==cell['parent_checkpoint_sha256']);assert cell['data_version']==parent['data_version']
        if cell['role']=='candidate':assert any(r['id']==cell['mlp_affine_parent_id'] and r['checkpoint_sha256']==cell['mlp_affine_parent_sha256'] and r['data_version']==cell['data_version'] for r in parents)
    torch.manual_seed(0);parent=controller_builder(16,'pd_factorized_linear');rng=torch.get_rng_state().clone()
    torch.manual_seed(0);model=build_model(16,'pd_factorized_affine');same(parent.state_dict(),model.state_dict());same(dict(parent.named_buffers()),dict(model.named_buffers()));same(torch.get_rng_state(),rng);assert model.parameter_count()==83664
    torch.manual_seed(0);mlp=controller_builder(16,'pd_mlp_affine')
    layer=next(model.ode_layers())[1].to(device).double();original=next(mlp.ode_layers())[1].to(device).double();P=layer.operator().detach();first,_,last=layer.t_func
    B=(.02*(last.weight@first.weight)).detach();a=(.6+.02*(last.weight@first.bias+last.bias)).detach();I=torch.eye(64,device=device,dtype=torch.double)
    assert float(torch.linalg.svdvals(B)[32])<1e-12
    for scale in [0.,1.,4.]:
        h=torch.linspace(-1.5,2,64,device=device,dtype=torch.double)*scale;d=a+B@h;q=d*h;r=P@q
        assert torch.allclose(layer.step_size(h[None])[0],d,atol=1e-12,rtol=1e-12)
        assert torch.allclose(layer(h[None])[0],h+r+.5*P@(d*r),atol=1e-11,rtol=1e-11)
        Jq=torch.diag(d)+torch.diag(h)@B;expected=I+P@Jq+.5*P@(torch.diag(d)@P@Jq+torch.diag(r)@B)
        actual=torch.autograd.functional.jacobian(lambda v:layer(v[None])[0],h);assert torch.allclose(actual,expected,atol=1e-11,rtol=1e-11)
    zero=torch.zeros(64,device=device,dtype=torch.double);weights=torch.linspace(-.8,.7,64,device=device,dtype=torch.double);v=torch.linspace(-.7,.9,64,device=device,dtype=torch.double);errors=[]
    for (name,layer),(_,original) in zip(model.ode_layers(),mlp.ode_layers()):
        layer=layer.to(device).double();original=original.to(device).double()
        error=lambda value:float(value.detach().abs().max())
        logit_error=error(layer.t_func(zero)-original.t_func(zero))
        logit_jac_error=error(torch.autograd.functional.jacobian(layer.t_func,zero)-torch.autograd.functional.jacobian(original.t_func,zero))
        jac_error=error(torch.autograd.functional.jacobian(lambda h:layer(h[None])[0],zero)-torch.autograd.functional.jacobian(lambda h:original(h[None])[0],zero))
        hessian_error=error(torch.autograd.functional.hessian(lambda h:(layer(h[None])[0]*weights).sum(),zero)-torch.autograd.functional.hessian(lambda h:(original(h[None])[0]*weights).sum(),zero))
        assert max(logit_error,logit_jac_error,jac_error,hessian_error)<2e-7
        fourth=sum(c*layer((t*v)[None])[0] for c,t in [(1,2),(-4,1),(6,0),(-4,-1),(1,-2)]);assert error(fourth)<1e-10
        stats=layer.activation_stats(v[None]);assert stats['clock_outer_saturated_fraction']==0 and stats['gate_transform']=='affine'
        errors.append(dict(layer=name,logit_error=logit_error,logit_jacobian_error=logit_jac_error,full_jacobian_error=jac_error,scalar_pullback_hessian_error=hessian_error,cubic_fourth_difference_max=error(fourth)))
    assert len(errors)==6
    row=dict(family='pd_factorized_affine',parameters=83664,common_tensors_buffers_rng_exact=True,rank_cap=32,complete_layer_derivative_passed=True,zero_anchor_layers=errors)
    return dict(passed=True,device=device,torch=torch.__version__,all_parent_data_versions_match=True,rows=[row])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--dataset');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False);grid=json.loads(Path(__file__).with_name('grid.json').read_text());rows=[];ds=a.dataset
        assert ds in ['gmm_ring8_2d','rotated_product_bimodal_16d']
        def run(runner,identifier,output,steps,log,resume=None):
            cmd=[sys.executable,str(runner),'--cell',identifier,'--data-root',a.data_root,'--output',str(output),'--device',a.device,'--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (root/log).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        for cell in [c for c in grid['cells'] if c['dataset']==ds]:
            tag=cell['id'];runner=Path(__file__).with_name('trial11.py');run(runner,cell['id'],root/'resume',8,tag+'-8.log')
            folder=root/'resume'/'professor_autoresearch_cycle11_20261103'/ds/cell['id']/'seed000';first=torch.load(folder/'checkpoints/step_0000008.pt',map_location='cpu',weights_only=False)
            if cell['role']=='anchor':
                number=2 if cell['parent_group']=='cycle2' else 3;parent_runner=Path(__file__).resolve().parents[1]/f'cycle{number}/trial{number}.py'
                run(parent_runner,cell['parent_id'],root/'parent',8,tag+'-parent.log');parent=torch.load(root/'parent'/f'professor_autoresearch_cycle{number}_20261103'/ds/cell['parent_id']/'seed000/checkpoints/step_0000008.pt',map_location='cpu',weights_only=False);numerical_equal(first,parent)
            run(runner,cell['id'],root/'resume',16,tag+'-resume16.log',folder/'checkpoints/step_0000008.pt');run(runner,cell['id'],root/'continuous',16,tag+'-continuous16.log')
            left=torch.load(folder/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False);right=torch.load(root/'continuous'/'professor_autoresearch_cycle11_20261103'/ds/cell['id']/'seed000/checkpoints/step_0000016.pt',map_location='cpu',weights_only=False);numerical_equal(left,right)
            rows.append(dict(cell=tag,family=cell['family'],native_resume16_exact_continuous16=True,parent8_exact=cell['role']=='anchor'))
        result.update(dataset=ds,native=rows)
    base.dump(a.output,result);print(json.dumps(result))
