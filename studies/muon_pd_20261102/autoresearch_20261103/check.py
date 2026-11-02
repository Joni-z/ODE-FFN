"""Native numerical and actual-trainer witnesses for the registered candidates."""
import argparse,io,json,subprocess,sys
from pathlib import Path
from trial import base,build_model,torch

def close(a,b,tol=0.): torch.testing.assert_close(a,b,atol=tol,rtol=tol)

def checks(device):
    base.verify_sources();torch.set_num_threads(1)
    anchors=[]
    for family in ['scalar_k2','vector_pd_k2','swiglu128','swiglu192']:
        for method,lr in [('adamw',.0003),('muon',.02)]:
            torch.manual_seed(0);a=base.build_model(16,family).to(device)
            torch.manual_seed(0);b=build_model(16,family).to(device)
            for name,p in a.state_dict().items():close(p,b.state_dict()[name])
            with torch.no_grad():a.output.weight.normal_(0,.1);b.output.weight.copy_(a.output.weight)
            oa,ob=[base.ProjectionOptimizer(m,method,lr) for m in [a,b]]
            for _ in range(3):
                x=torch.randn(32,16,device=device);t=torch.rand(32,1,device=device)
                for m,opt in [(a,oa),(b,ob)]:
                    opt.zero_grad(set_to_none=True);m(x,t).square().mean().backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                for name,p in a.named_parameters():close(p,dict(b.named_parameters())[name])
            anchors.append(family+'/'+method)
    candidates=[]
    for family in ['pd_linear_sigmoid','pd_mlp_affine']:
        torch.manual_seed(0);old=base.build_model(16,'vector_pd_k2')
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(0);model=build_model(16,family)
        close(torch.get_rng_state(),expected_rng)
        for name,p in old.named_parameters():
            if '.t_func.' not in name:close(p,dict(model.named_parameters())[name])
        layer=next(model.ode_layers())[1].to(device).double()
        previous=next(old.ode_layers())[1].to(device).double()
        zero=torch.zeros(64,device=device,dtype=torch.double)
        if family=='pd_linear_sigmoid':
            close(layer.t_func(zero),previous.t_func(zero),2e-6)
            close(torch.autograd.functional.jacobian(layer.t_func,zero),torch.autograd.functional.jacobian(previous.t_func,zero),2e-6)
        h=torch.linspace(-2,2,64,device=device,dtype=torch.double)
        def dense(x):
            d=torch.diag(layer.step_size(x[None])[0]);a=layer.operator()@d
            return x+a@x+a@(a@x)/2
        close(layer(h[None])[0],dense(h),1e-11)
        jac=torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h)
        close(jac,torch.autograd.functional.jacobian(dense,h),1e-10)
        v=torch.cos(h);epsilon=1e-5
        close(jac@v,(dense(h+epsilon*v)-dense(h-epsilon*v))/(2*epsilon),1e-8)
        stats=layer.activation_stats(h[None]);assert len(stats['conditional_generator_spectra'])==1
        close(torch.tensor(stats['conditional_generator_spectra'][0]['singular_values'],device=device),torch.linalg.svdvals(layer.operator()@torch.diag(layer.step_size(h[None])[0])).float(),1e-5)
        # Physically save model, optimizer and device RNG and reproduce next update.
        for method,lr in [('adamw',.0003),('muon',.02)]:
            torch.manual_seed(0);m=build_model(16,family).to(device);opt=base.ProjectionOptimizer(m,method,lr)
            rng=torch.Generator(device=device).manual_seed(100000)
            def step():
                x=torch.randn(32,16,device=device,generator=rng);t=torch.rand(32,1,device=device,generator=rng)
                opt.zero_grad(set_to_none=True);loss=(m(x,t)-torch.sin(x)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();assert torch.isfinite(loss);return loss.item()
            step();step();state=base.train.capture_state(m,opt,rng,2,'cfg','data',{'seed':0})
            buf=io.BytesIO();torch.save(state,buf);expected_loss=step();expected={n:p.detach().clone() for n,p in m.named_parameters()}
            buf.seek(0);restored=torch.load(buf,map_location=device,weights_only=False);base.train.restore_model(m,restored);opt.load_state_dict(restored['optimizer']);rng.set_state(restored['data_rng'].cpu())
            assert step()==expected_loss
            for n,p in m.named_parameters():close(p,expected[n])
        candidates.append(dict(family=family,parameters=model.parameter_count(),dense_and_full_derivative=True,physical_resume=True))
    return dict(passed=True,device=device,torch=torch.__version__,anchor_exact_three_updates=anchors,candidates=candidates)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--smoke-output');a=p.parse_args()
    result=checks(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False)
        grid=json.loads(Path(__file__).with_name('grid.json').read_text())
        ds='rotated_product_bimodal_16d' if (Path(a.data_root)/'product16_v1').exists() and not (Path(a.data_root)/'v1/gmm_ring8_2d').exists() else 'gmm_ring8_2d'
        for family in ['pd_linear_sigmoid','pd_mlp_affine']:
            cell=next(c for c in grid['cells'] if c['dataset']==ds and c['family']==family and c['optimizer']=='muon' and c['projection_lr']==.006)
            command=[sys.executable,str(Path(__file__).with_name('trial.py')),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device',a.device,'--steps','8']
            with (root/(family+'.log')).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            folder=root/'professor_autoresearch_20261103'/ds/cell['id']/'seed000'
            assert (folder/'ADAPTER_COMPLETE').exists()
            command[-1]='16';command+=['--resume',str(folder/'checkpoints/step_0000008.pt')]
            with (root/(family+'-resume.log')).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert (folder/'checkpoints/step_0000016.pt').exists()
        result['actual_native_resume_to_step']=16
    base.dump(a.output,result);print(json.dumps(result))
