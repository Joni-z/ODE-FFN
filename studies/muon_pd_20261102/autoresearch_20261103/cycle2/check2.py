"""Native witnesses for direct D and factorized-linear gate counterfactuals."""
import argparse,io,json,subprocess,sys
from pathlib import Path
from trial2 import base,build_model,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check import checks,close


def verify(device):
    result=checks(device)
    result['cycle2']=[]
    for family in ['pd_factorized_linear','pd_direct_raw','pd_direct_bounded']:
        torch.manual_seed(0);m=build_model(16,family)
        state=torch.get_rng_state().clone()
        torch.manual_seed(0);parent=base.build_model(16,'vector_pd_k2')
        close(torch.get_rng_state(),state)
        for name,p in parent.named_parameters():
            if '.t_func.' not in name:close(p,dict(m.named_parameters())[name])
        if family=='pd_factorized_linear':
            assert m.parameter_count()==parent.parameter_count()
            torch.manual_seed(0);affine=build_model(16,'pd_linear_sigmoid')
            for (_,left),(_,right) in zip(m.ode_layers(),affine.ode_layers()):
                x=torch.randn(64,64);close(left.step_size(x),right.step_size(x),2e-6)
        else:
            assert not any('.t_func.' in n for n,_ in m.named_parameters())
        layer=next(m.ode_layers())[1].to(device).double()
        for h in [torch.zeros(64,device=device,dtype=torch.double),torch.linspace(-2,2,64,device=device,dtype=torch.double)]:
            def dense(x):
                d=layer.step_size(x[None])[0];a=layer.operator()@torch.diag(d)
                return x+a@x+a@(a@x)/2
            close(layer(h[None])[0],dense(h),1e-11)
            jac=torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h)
            close(jac,torch.autograd.functional.jacobian(dense,h),1e-10)
            eps=1e-5;v=torch.cos(h)
            close(jac@v,(dense(h+eps*v)-dense(h-eps*v))/(2*eps),2e-8)
            stats=layer.activation_stats(h[None]);assert 'conditional_generator_spectra' in stats
            assert all(torch.isfinite(torch.tensor(x['singular_values'])).all() for x in stats['conditional_generator_spectra'])
            if family=='pd_direct_raw':
                p=layer.operator();explicit=h+p@(h*h)+.5*p@(h*(p@(h*h)))
                close(dense(h),explicit,1e-11)
        for method,lr in [('adamw',.0003),('muon',.02)]:
            torch.manual_seed(0);model=build_model(16,family).to(device);opt=base.ProjectionOptimizer(model,method,lr);rng=torch.Generator(device=device).manual_seed(100000)
            def update():
                x=torch.randn(32,16,device=device,generator=rng);t=torch.rand(32,1,device=device,generator=rng)
                opt.zero_grad(set_to_none=True);loss=(model(x,t)-torch.sin(x)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();assert torch.isfinite(loss);return loss.item()
            update();update();buf=io.BytesIO();torch.save(base.train.capture_state(model,opt,rng,2,'cfg','data',{'seed':0}),buf)
            expected_loss=update();expected={n:p.detach().clone() for n,p in model.named_parameters()}
            buf.seek(0);s=torch.load(buf,map_location=device,weights_only=False);base.train.restore_model(model,s);opt.load_state_dict(s['optimizer']);rng.set_state(s['data_rng'].cpu());assert update()==expected_loss
            for n,p in model.named_parameters():close(p,expected[n])
        result['cycle2'].append(dict(family=family,parameters=m.parameter_count(),zero_and_signed_dense_full_derivatives=True,physical_resume=True))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False)
        grid=json.loads(Path(__file__).with_name('grid.json').read_text());ds='rotated_product_bimodal_16d' if (Path(a.data_root)/'product16_v1').exists() and not (Path(a.data_root)/'v1/gmm_ring8_2d').exists() else 'gmm_ring8_2d'
        for family in ['pd_factorized_linear','pd_direct_raw','pd_direct_bounded']:
            cell=next(c for c in grid['cells'] if c['dataset']==ds and c['family']==family and c['optimizer']=='muon' and c['projection_lr']==.006)
            cmd=[sys.executable,str(Path(__file__).with_name('trial2.py')),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device',a.device,'--steps','8']
            with (root/(family+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            folder=root/'professor_autoresearch_cycle2_20261103'/ds/cell['id']/'seed000';assert (folder/'ADAPTER_COMPLETE').exists()
            cmd[-1]='16';cmd+=['--resume',str(folder/'checkpoints/step_0000008.pt')]
            with (root/(family+'-resume.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert (folder/'checkpoints/step_0000016.pt').exists()
        result['actual_native_resume_to_step']=16
    base.dump(a.output,result);print(json.dumps(result))
