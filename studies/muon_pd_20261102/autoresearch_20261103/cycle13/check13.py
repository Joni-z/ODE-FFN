"""Exact copied control, D-only calculus and real native entry/resume checks."""
import argparse,csv,inspect,json,subprocess,sys
from pathlib import Path
from trial13 import base,build_model,previous_builder,torch,projection_parameters,native_projection_parameters
from ordinary import cubic_flow,FreeISRU
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle6'))
from prefix import same,numerical_equal

HERE=Path(__file__).resolve().parent
ROOT=base.HERE.parents[1]
PARENTS={'cycle4':('autoresearch_20261103/cycle4/trial4.py','professor_autoresearch_cycle4_20261103'),
         'cycle3':('autoresearch_20261103/cycle3/trial3.py','professor_autoresearch_cycle3_20261103'),
         'old':('run.py','muon_pd_20261102')}

def verify(device):
    import numpy as np
    from synthetic.data import suite
    base.verify_sources();torch.set_num_threads(1)
    provenance=json.loads((HERE/'provenance.json').read_text())
    assert base.digest(ROOT/provenance['source'])==provenance['source_sha256']
    import hashlib
    for obj in [cubic_flow,FreeISRU]:
        assert hashlib.sha256(inspect.getsource(obj).rstrip().encode()).hexdigest()==provenance['copied_definitions'][obj.__name__]
    grid=json.loads((HERE/'grid.json').read_text());assert len(grid['cells'])==18
    assert len({c['id'] for c in grid['cells']})==18
    paths=['docs/autoresearch/professor-20261103/cycle4-results/endpoints.csv','docs/autoresearch/professor-20261103/cycle3-results/endpoints.csv','docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv']
    parents=[r for f in paths for r in csv.DictReader((ROOT/f).open())]
    for cell in grid['cells']:
        if 'parent_id' in cell:
            parent=next(r for r in parents if r['id']==cell['parent_id'] and r['checkpoint_sha256']==cell['parent_checkpoint_sha256'])
            assert parent['data_version']==cell['data_version'] and parent['dataset']==cell['dataset']
    rows=[]
    for dim in [2,16]:
        torch.manual_seed(0);parent=previous_builder(dim,'pd_constant_k1');rng=torch.get_rng_state().clone()
        for family in ['pd_isru_k1','free_isru','pd_constant_k1','swiglu64','swiglu128','swiglu192']:
            torch.manual_seed(0);model=build_model(dim,family);same(torch.get_rng_state(),rng)
            if family=='pd_isru_k1':same(parent.state_dict(),model.state_dict());same(dict(parent.named_buffers()),dict(model.named_buffers()))
            else:
                source=dict(parent.named_parameters())
                for name,p in model.named_parameters():
                    if '.ffn.' not in name:same(p,source[name])
                same(dict(parent.named_buffers()),dict(model.named_buffers()))
            selected=projection_parameters(model)
            if family=='free_isru':assert len(selected)==6 and all(n.endswith(('.matrix1','.matrix2')) for n,_ in selected)
            else:assert [n for n,_ in selected]==[n for n,_ in native_projection_parameters(model)]
            assert len({id(p) for _,p in selected})==len(selected)
            if family in ['pd_isru_k1','pd_constant_k1','free_isru']:assert model.parameter_count()==(56706 if dim==2 else 58512)
            rows.append(dict(dim=dim,family=family,parameters=model.parameter_count(),matrices=len(selected),common_surrounding_buffers_rng_exact=True))
    torch.manual_seed(0);model=build_model(16,'pd_isru_k1').to(device).double()
    layer=next(model.ode_layers())[1];P=layer.operator().detach();I=torch.eye(64,device=device,dtype=torch.double)
    errors=[]
    for scale in [0.,.1,1.,1000.]:
        h=torch.linspace(-1.5,2.,64,device=device,dtype=torch.double)*scale
        q=h*layer.step_size(h[None])[0];expected=h+P@q;actual=layer(h[None])[0]
        assert torch.allclose(actual,expected,rtol=1e-11,atol=1e-11)
        assert torch.equal(q,-(-h*layer.step_size((-h)[None])[0]))
        assert float(q.abs().max())<=1/1.2**.5+1e-12
        jac=torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h)
        analytical=I+P@(torch.diag((1+1.2*h.square()).pow(-1.5)))
        assert torch.allclose(jac,analytical,rtol=1e-10,atol=1e-10)
        errors.append(float((jac-analytical).abs().max()))
    z=torch.zeros(64,device=device,dtype=torch.double)
    hessian=torch.autograd.functional.hessian(lambda x:layer(x[None])[0,7],z)
    assert torch.equal(hessian,torch.zeros_like(hessian))
    scalar=torch.zeros((),device=device,dtype=torch.double,requires_grad=True)
    q=scalar*torch.rsqrt(1+1.2*scalar.square())
    for _ in range(3):q=torch.autograd.grad(q,scalar,create_graph=True)[0]
    assert abs(float(q.detach())+3.6)<1e-12
    assert 'conditional_generator_spectra' in layer.activation_stats(torch.ones(1,64,device=device,dtype=torch.double))
    torch.manual_seed(0);free=build_model(16,'free_isru').to(device).double();module=free.blocks[0].ffn
    x=torch.linspace(-2,2,64,device=device,dtype=torch.double)
    u=module.matrix1@x;explicit=module.matrix2@(u*torch.rsqrt(1+1.2*u.square()))
    assert torch.allclose(module(x),explicit,rtol=1e-12,atol=1e-12)
    jac=torch.autograd.functional.jacobian(module,x)
    expected=module.matrix2@torch.diag((1+1.2*u.square()).pow(-1.5))@module.matrix1
    assert torch.allclose(jac,expected,rtol=1e-11,atol=1e-11)
    assert module.matrix1 is not module.matrix2 and torch.equal(module.matrix1,module.matrix2)
    # Native formula, not a fitted surrogate or a new evaluation metric.
    law=suite()['rotated_product_bimodal_16d']
    x=np.linspace(-3,3,16*33).reshape(33,16);t=np.linspace(0,1,33)[:,None]
    latent=x@law.rotation;v=(1-t)**2+t*t*law.variances
    a=(t*law.variances-(1-t))/v;b=law.locations*(1-t)/v;c=t*law.locations/v
    derived=(a*latent+b*np.tanh(c*latent))@law.rotation.T
    error=float(np.max(np.abs(derived-law.velocity(x,t))))
    assert error<1e-12
    return dict(passed=True,device=device,torch=torch.__version__,rows=rows,complete_jacobian_max_error=max(errors),zero_hessian_exact=True,third_feature_derivative=-3.6,copy_source_exact=True,oracle_max_error=error,all_parent_records_match=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--dataset');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False);grid=json.loads((HERE/'grid.json').read_text());rows=[]
        def run(runner,ident,out,steps,log,resume=None):
            cmd=[sys.executable,str(runner),'--cell',ident,'--data-root',a.data_root,'--output',str(out),'--device',a.device,'--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (root/log).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        for cell in [c for c in grid['cells'] if c['dataset']==a.dataset]:
            base.verify_data(a.data_root,cell)
            tag=cell['id'];runner=HERE/'trial13.py';exp='professor_autoresearch_cycle13_20261103'
            run(runner,tag,root/'resume',8,tag+'-8.log')
            folder=root/'resume'/exp/a.dataset/tag/'seed000'
            first=torch.load(folder/'checkpoints/step_0000008.pt',map_location='cpu',weights_only=False)
            if cell['role']=='anchor':
                parent_runner,parent_exp=PARENTS[cell['parent_group']]
                run(base.HERE/parent_runner,cell['parent_id'],root/'parent',8,tag+'-parent.log')
                old=torch.load(root/'parent'/parent_exp/a.dataset/cell['parent_id']/'seed000/checkpoints/step_0000008.pt',map_location='cpu',weights_only=False);numerical_equal(first,old)
            run(runner,tag,root/'resume',16,tag+'-resume16.log',folder/'checkpoints/step_0000008.pt')
            run(runner,tag,root/'continuous',16,tag+'-continuous16.log')
            left=torch.load(folder/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False)
            right=torch.load(root/'continuous'/exp/a.dataset/tag/'seed000/checkpoints/step_0000016.pt',map_location='cpu',weights_only=False);numerical_equal(left,right)
            rows.append(dict(cell=tag,native_resume16_exact_continuous16=True,parent8_exact=cell['role']=='anchor'))
            base.dump(root/'progress.json',dict(completed=len(rows),rows=rows))
        assert len(rows)==9;result.update(dataset=a.dataset,native=rows)
    base.dump(a.output,result);print(json.dumps(result))
