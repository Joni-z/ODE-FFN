"""Native no-op equality and real constrained8-to16 resume witness."""
import argparse,copy,csv,json,subprocess,sys
from pathlib import Path
from trial8 import base,build_model,RelaxedOptimizer,variants,torch,STRENGTH
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle6'))
from prefix import same,numerical_equal

def verify(device):
    base.verify_sources();torch.set_num_threads(1);rows=[]
    parents=[]
    repo=base.HERE.parents[1]
    for f in ['docs/autoresearch/professor-20261103/cycle3-results/endpoints.csv','docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv']:
        parents+=list(csv.DictReader((repo/f).open()))
    grid=json.loads(Path(__file__).with_name('grid.json').read_text())
    for cell in grid['cells']:
        parent=next(r for r in parents if r['id']==cell['parent_id'] and r['checkpoint_sha256']==cell['parent_checkpoint_sha256'])
        assert cell['data_version']==parent['data_version'],(cell['id'],'parent data version differs')
    for family in ['scalar_k2','pd_direct_raw_k1']:
        torch.manual_seed(0);model=build_model(16,family).to(device)
        reference=copy.deepcopy(model);native=base.ProjectionOptimizer(reference,'muon',.006)
        cell=dict(optimizer='muon',projection_lr=.006,relaxation='none')
        wrapped=RelaxedOptimizer(model,cell)
        rng=torch.Generator(device=device).manual_seed(123)
        for step in range(1,17):
            x=torch.randn(32,16,device=device,generator=rng);t=torch.rand(32,1,device=device,generator=rng)
            for m,opt in [(model,wrapped),(reference,native)]:
                opt.zero_grad(set_to_none=True);loss=(m(x,t)-torch.sin(x)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            same(model.state_dict(),reference.state_dict());same(wrapped.state_dict(),native.state_dict())
        # Real momentum step plus each relaxation, with direct total-delta checks.
        for mode in ['shape','scale']:
            constrained=copy.deepcopy(reference);plain=copy.deepcopy(reference)
            copt=RelaxedOptimizer(constrained,dict(cell,relaxation=mode));popt=base.ProjectionOptimizer(plain,'muon',.006)
            # deepcopy state tensors to avoid optimizer-state aliases in the witness.
            copt.load_state_dict(copy.deepcopy(native.state_dict()));popt.load_state_dict(copy.deepcopy(native.state_dict()))
            copt.count=popt.count=19
            before={n:p.detach().clone() for n,p in copt.selected}
            for m,opt in [(constrained,copt),(plain,popt)]:
                opt.zero_grad(set_to_none=True);loss=(m(x,t)-torch.sin(x)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            for (n,p),(n2,q) in zip(copt.selected,popt.selected):
                assert n==n2;expected=variants(q,STRENGTH)[mode=='scale']
                assert torch.equal(p,expected)
                native_delta=q.double()-before[n].double();relax_delta=p.double()-q.double();total=p.double()-before[n].double()
                assert torch.allclose(total,native_delta+relax_delta,rtol=0,atol=1e-14)
        rows.append(dict(family=family,native_none_16_steps_exact=True,post_step_shape_scale_matches_formula=True))
    return dict(passed=True,device=device,torch=torch.__version__,rows=rows,strength=STRENGTH,all_parent_data_versions_match=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--dataset');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False)
        grid=json.loads(Path(__file__).with_name('grid.json').read_text());rows=[]
        assert a.dataset in ['gmm_ring8_2d','rotated_product_bimodal_16d'];ds=a.dataset
        def run(runner,cell,out,steps,log,resume=None):
            cmd=[sys.executable,str(runner),'--cell',cell,'--data-root',a.data_root,'--output',str(out),'--device',a.device,'--steps',str(steps)]
            if resume:cmd+=['--resume',str(resume)]
            with (root/log).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        for cell in [c for c in grid['cells'] if c['dataset']==ds]:
            family=cell['family'];mode=cell['relaxation'];tag=family+'-'+mode;runner=Path(__file__).with_name('trial8.py')
            run(runner,cell['id'],root/'resume',8,tag+'-8.log')
            folder=root/'resume'/'professor_autoresearch_cycle8_20261103'/ds/cell['id']/'seed000'
            first=torch.load(folder/'checkpoints/step_0000008.pt',map_location='cpu',weights_only=False)
            if mode=='none':
                parent_runner=base.HERE/'run.py' if family=='scalar_k2' else Path(__file__).resolve().parents[1]/'cycle3/trial3.py'
                run(parent_runner,cell['parent_id'],root/'parent',8,tag+'-parent.log')
                exp='muon_pd_20261102' if family=='scalar_k2' else 'professor_autoresearch_cycle3_20261103'
                parent=torch.load(root/'parent'/exp/ds/cell['parent_id']/'seed000/checkpoints/step_0000008.pt',map_location='cpu',weights_only=False)
                numerical_equal(first,parent)
            run(runner,cell['id'],root/'resume',16,tag+'-resume16.log',folder/'checkpoints/step_0000008.pt')
            run(runner,cell['id'],root/'continuous',16,tag+'-continuous16.log')
            resumed=torch.load(folder/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False)
            continuous=torch.load(root/'continuous'/'professor_autoresearch_cycle8_20261103'/ds/cell['id']/'seed000/checkpoints/step_0000016.pt',map_location='cpu',weights_only=False)
            numerical_equal(resumed,continuous)
            events=[json.loads(line) for line in (folder/'update_spectra.jsonl').read_text().splitlines()]
            boundary=next(r for r in events if r['step']==10)
            assert boundary['relaxation_applied']==(mode!='none')
            rows.append(dict(family=family,relaxation=mode,native_resume16_exact_continuous16=True,native_eight_step_parent_exact=mode=='none',crossed_step10_relaxation=True))
        result.update(dataset=ds,native=rows)
    base.dump(a.output,result);print(json.dumps(result))
