"""Dense feature derivatives and unchanged native trainer/resume witnesses."""
import argparse,io,json,subprocess,sys
from pathlib import Path
from trial4 import base,build_model,previous_builder,torch,FAMILIES
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from check3 import verify as verify_previous,close
from trial3 import previous_builder as cycle2_builder


def verify(device):
    result=verify_previous(device);result['cycle4']=[]
    for family in FAMILIES:
        torch.manual_seed(0);parent=previous_builder(16,'pd_direct_raw_k1');after=torch.get_rng_state().clone()
        torch.manual_seed(0);model=build_model(16,family);close(torch.get_rng_state(),after)
        assert model.parameter_count()==58512
        for n,p in model.named_parameters():close(p,dict(parent.named_parameters())[n])
        for n,b in model.named_buffers():close(b,dict(parent.named_buffers())[n])
        layer=next(model.ode_layers())[1].to(device).double()
        for h in [torch.zeros(64,device=device,dtype=torch.double),torch.linspace(-2,2,64,device=device,dtype=torch.double)]:
            p=layer.operator()
            if family=='pd_constant_k1':feature=h;derivative=torch.ones_like(h)
            elif family=='pd_direct_shift_k1':feature=h*h+h;derivative=2*h+1
            else:
                q=torch.sigmoid(h/10);d=.2+.8*q;feature=h*d;derivative=d+h*.08*q*(1-q)
            expected=h+p@feature;jac=torch.eye(64,device=device,dtype=torch.double)+p*derivative[None,:]
            close(layer(h[None])[0],expected,1e-11)
            close(torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h),jac,1e-10)
            stats=layer.activation_stats(h[None]);assert 'conditional_generator_spectra' in stats
        if family=='pd_direct_bounded_k1':
            torch.manual_seed(0);inherited=cycle2_builder(16,'pd_direct_bounded')
            for _,block in inherited.ode_layers():block.orders=1
            torch.manual_seed(0);same=build_model(16,family)
            x=torch.linspace(-2,2,64)[None].repeat(4,1);t=torch.linspace(.1,.9,4)[:,None]
            close(same(x[:,:16],t),inherited(x[:,:16],t),0.)
        for method,lr in [('adamw',.0003),('muon',.02)]:
            torch.manual_seed(0);m=build_model(16,family).to(device);opt=base.ProjectionOptimizer(m,method,lr);rng=torch.Generator(device=device).manual_seed(100000)
            def update():
                x=torch.randn(32,16,device=device,generator=rng);t=torch.rand(32,1,device=device,generator=rng)
                opt.zero_grad(set_to_none=True);loss=(m(x,t)-torch.sin(x)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();assert torch.isfinite(loss);return loss.item()
            update();update();buf=io.BytesIO();torch.save(base.train.capture_state(m,opt,rng,2,'cfg','data',{'seed':0}),buf)
            expected_loss=update();expected={n:p.detach().clone() for n,p in m.named_parameters()}
            buf.seek(0);s=torch.load(buf,map_location=device,weights_only=False);base.train.restore_model(m,s);opt.load_state_dict(s['optimizer']);rng.set_state(s['data_rng'].cpu());assert update()==expected_loss
            for n,p in m.named_parameters():close(p,expected[n])
        result['cycle4'].append(dict(family=family,parameters=model.parameter_count(),dense_full_derivatives=True,physical_resume=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False);grid=json.loads(Path(__file__).with_name('grid.json').read_text())
        ds='rotated_product_bimodal_16d' if (Path(a.data_root)/'product16_v1').exists() and not (Path(a.data_root)/'v1/gmm_ring8_2d').exists() else 'gmm_ring8_2d'
        for family in FAMILIES:
            cell=next(c for c in grid['cells'] if c['dataset']==ds and c['family']==family and c['optimizer']=='muon' and c['projection_lr']==.006)
            cmd=[sys.executable,str(Path(__file__).with_name('trial4.py')),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device',a.device,'--steps','8']
            with (root/(family+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            folder=root/'professor_autoresearch_cycle4_20261103'/ds/cell['id']/'seed000';assert (folder/'ADAPTER_COMPLETE').exists()
            cmd[-1]='16';cmd+=['--resume',str(folder/'checkpoints/step_0000008.pt')]
            with (root/(family+'-resume.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert (folder/'checkpoints/step_0000016.pt').exists()
        result['actual_native_resume_to_step']=16
    base.dump(a.output,result);print(json.dumps(result))
