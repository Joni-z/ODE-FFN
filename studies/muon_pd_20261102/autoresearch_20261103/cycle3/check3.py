"""Native K1 algebra, smaller author-SwiGLU and actual trainer resume witnesses."""
import argparse,io,json,subprocess,sys
from pathlib import Path
from trial3 import base,build_model,previous_builder,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check import checks,close


def verify(device):
    result=checks(device);result['cycle3']=[]
    for family in ['scalar_k1','pd_direct_raw_k1','swiglu64']:
        torch.manual_seed(0);parent=(previous_builder(16,'pd_direct_raw') if family=='pd_direct_raw_k1' else base.build_model(16,'scalar_k2'));rng_after=torch.get_rng_state().clone()
        torch.manual_seed(0);model=build_model(16,family)
        close(torch.get_rng_state(),rng_after)
        for n,p in model.named_parameters():
            if family in ['scalar_k1','pd_direct_raw_k1'] or '.ffn.' not in n:close(p,dict(parent.named_parameters())[n])
        for n,b in model.named_buffers():close(b,dict(parent.named_buffers())[n])
        if family in ['scalar_k1','pd_direct_raw_k1']:
            layer=next(model.ode_layers())[1].to(device).double()
            h=torch.linspace(-2,2,64,device=device,dtype=torch.double)
            dense=lambda x:x+layer.operator()@(layer.step_size(x[None])[0]*x)
            close(layer(h[None])[0],dense(h),1e-11)
            close(torch.autograd.functional.jacobian(lambda x:layer(x[None])[0],h),torch.autograd.functional.jacobian(dense,h),1e-10)
        else:
            assert model.parameter_count()==58572
            block=model.blocks[0].ffn.to(device).double()
            assert block.w12.weight.shape==(84,64) and block.w3.weight.shape==(64,42)
            def dense(x):
                a=torch.nn.functional.linear(x,block.w12.weight,block.w12.bias);g,v=a.chunk(2,-1)
                return torch.nn.functional.linear(torch.nn.functional.silu(g)*v,block.w3.weight,block.w3.bias)
            h=torch.linspace(-2,2,64,device=device,dtype=torch.double)
            close(block(h),dense(h),1e-12)
            close(torch.autograd.functional.jacobian(block,h),torch.autograd.functional.jacobian(dense,h),1e-12)
        for method,lr in [('adamw',.001),('muon',.02)]:
            torch.manual_seed(0);m=build_model(16,family).to(device);opt=base.ProjectionOptimizer(m,method,lr);rng=torch.Generator(device=device).manual_seed(100000)
            def update():
                x=torch.randn(32,16,device=device,generator=rng);t=torch.rand(32,1,device=device,generator=rng)
                opt.zero_grad(set_to_none=True);loss=(m(x,t)-torch.sin(x)).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();assert torch.isfinite(loss);return loss.item()
            update();update();buf=io.BytesIO();torch.save(base.train.capture_state(m,opt,rng,2,'cfg','data',{'seed':0}),buf)
            expected_loss=update();expected={n:p.detach().clone() for n,p in m.named_parameters()}
            buf.seek(0);s=torch.load(buf,map_location=device,weights_only=False);base.train.restore_model(m,s);opt.load_state_dict(s['optimizer']);rng.set_state(s['data_rng'].cpu());assert update()==expected_loss
            for n,p in m.named_parameters():close(p,expected[n])
        result['cycle3'].append(dict(family=family,parameters=model.parameter_count(),dense_full_derivatives=True,physical_resume=True))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False);grid=json.loads(Path(__file__).with_name('grid.json').read_text())
        ds='rotated_product_bimodal_16d' if (Path(a.data_root)/'product16_v1').exists() and not (Path(a.data_root)/'v1/gmm_ring8_2d').exists() else 'gmm_ring8_2d'
        for family in ['scalar_k1','pd_direct_raw_k1','swiglu64']:
            cell=next(c for c in grid['cells'] if c['dataset']==ds and c['family']==family and c['optimizer']=='muon' and c['projection_lr']==.006)
            cmd=[sys.executable,str(Path(__file__).with_name('trial3.py')),'--cell',cell['id'],'--data-root',a.data_root,'--output',str(root),'--device',a.device,'--steps','8']
            with (root/(family+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            folder=root/'professor_autoresearch_cycle3_20261103'/ds/cell['id']/'seed000';assert (folder/'ADAPTER_COMPLETE').exists()
            cmd[-1]='16';cmd+=['--resume',str(folder/'checkpoints/step_0000008.pt')]
            with (root/(family+'-resume.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert (folder/'checkpoints/step_0000016.pt').exists()
        result['actual_native_resume_to_step']=16
    base.dump(a.output,result);print(json.dumps(result))
