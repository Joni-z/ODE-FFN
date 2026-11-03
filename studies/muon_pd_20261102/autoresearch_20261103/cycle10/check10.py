"""D-only identities, complete derivatives and actual native entry/resume."""
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
from trial10 import base,build_model,previous_builder,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle6'))
from prefix import same,numerical_equal

def verify(device):
    base.verify_sources();torch.set_num_threads(1);parents=[]
    for f in ['docs/autoresearch/professor-20261103/cycle3-results/endpoints.csv','docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv']:
        parents+=list(csv.DictReader((base.HERE.parents[1]/f).open()))
    grid=json.loads(Path(__file__).with_name('grid.json').read_text());assert len(grid['cells'])==12
    for cell in grid['cells']:
        parent=next(r for r in parents if r['id']==cell['parent_id'] and r['checkpoint_sha256']==cell['parent_checkpoint_sha256'])
        assert cell['data_version']==parent['data_version']
    rows=[]
    for family in ['pd_radial_k1','pd_scaled_raw_k1']:
        torch.manual_seed(0);parent=previous_builder(16,'pd_direct_raw_k1');rng=torch.get_rng_state().clone()
        torch.manual_seed(0);model=build_model(16,family);same(parent.state_dict(),model.state_dict());same(dict(parent.named_buffers()),dict(model.named_buffers()));same(torch.get_rng_state(),rng);assert model.parameter_count()==58512
        layer=next(model.ode_layers())[1].to(device).double();P=layer.operator().detach();w=64;I=torch.eye(w,device=device,dtype=torch.double)
        for scale in [0.,.1,1.,1000.]:
            h=torch.linspace(-1.5,2,64,device=device,dtype=torch.double)*scale
            rate=layer.step_size(h[None])[0];q=h*rate
            if family=='pd_radial_k1':
                radius=(1+h.square().mean()).sqrt();u=h/radius
                Jq=torch.diag(u)@(2*I-u[:,None]*u[None,:]/w)
                assert float(torch.linalg.matrix_norm(Jq,ord=2))<=2*math.sqrt(w)+1e-10
                assert torch.allclose(q*radius,h*h,rtol=1e-11,atol=1e-12)
                assert torch.allclose(q,(-h)*layer.step_size((-h)[None])[0],rtol=0,atol=0)
                assert float(rate.abs().max())<=math.sqrt(w)+1e-10
            else:Jq=math.sqrt(2)*torch.diag(h)
            expected=h+P@q;actual=layer(h[None])[0]
            assert torch.allclose(actual,expected,rtol=1e-10,atol=1e-10)
            jac=torch.autograd.functional.jacobian(lambda v:layer(v[None])[0],h)
            assert torch.allclose(jac,I+P@Jq,rtol=1e-10,atol=1e-10)
        zero=torch.zeros(64,device=device,dtype=torch.double)
        hessian=torch.autograd.functional.hessian(lambda v:(v*layer.step_size(v[None])[0])[7],zero)
        expected=torch.zeros_like(hessian);expected[7,7]=2 if family=='pd_radial_k1' else math.sqrt(2)
        assert torch.allclose(hessian,expected,atol=1e-12,rtol=0)
        unit=torch.ones(64,device=device,dtype=torch.double)
        assert torch.allclose(unit*layer.step_size(unit[None])[0],unit.square()/math.sqrt(2),atol=1e-12,rtol=0)
        stats=layer.activation_stats(unit[None]);assert 'conditional_generator_spectra' in stats and 'clock_outer_saturated_fraction' not in stats
        rows.append(dict(family=family,parameters=58512,common_tensors_buffers_rng_exact=True,complete_layer_derivative_exact=True,zero_anchor_quadratic_coefficient=True,quadratic_hessian_entry=float(expected[7,7]),unit_rms_static_match=True))
    return dict(passed=True,device=device,torch=torch.__version__,all_parent_data_versions_match=True,rows=rows)

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
            tag=cell['family'];runner=Path(__file__).with_name('trial10.py')
            run(runner,cell['id'],root/'resume',8,tag+'-8.log')
            folder=root/'resume'/'professor_autoresearch_cycle10_20261103'/ds/cell['id']/'seed000'
            first=torch.load(folder/'checkpoints/step_0000008.pt',map_location='cpu',weights_only=False)
            if cell['role']=='anchor':
                parent_runner=base.HERE/'run.py' if cell['parent_group']=='old' else Path(__file__).resolve().parents[1]/'cycle3/trial3.py'
                run(parent_runner,cell['parent_id'],root/'parent',8,tag+'-parent.log')
                experiment='muon_pd_20261102' if cell['parent_group']=='old' else 'professor_autoresearch_cycle3_20261103'
                parent=torch.load(root/'parent'/experiment/ds/cell['parent_id']/'seed000/checkpoints/step_0000008.pt',map_location='cpu',weights_only=False);numerical_equal(first,parent)
            run(runner,cell['id'],root/'resume',16,tag+'-resume16.log',folder/'checkpoints/step_0000008.pt')
            run(runner,cell['id'],root/'continuous',16,tag+'-continuous16.log')
            left=torch.load(folder/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False)
            right=torch.load(root/'continuous'/'professor_autoresearch_cycle10_20261103'/ds/cell['id']/'seed000/checkpoints/step_0000016.pt',map_location='cpu',weights_only=False);numerical_equal(left,right)
            rows.append(dict(family=tag,native_resume16_exact_continuous16=True,parent8_exact=cell['role']=='anchor'))
        result.update(dataset=ds,native=rows)
    base.dump(a.output,result);print(json.dumps(result))
