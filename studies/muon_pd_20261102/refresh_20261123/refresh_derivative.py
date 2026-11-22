"""Measure the extra local derivative obtained by refreshing the fitted ISRU diagonal."""
import json
from pathlib import Path
import sys
from odd_trial import base, build_model, torch
from evaluate_screen import assets, ev

@torch.no_grad()
def run(work,out):
    torch.set_num_threads(1)
    ev.build_model=build_model
    result=[]
    for s in json.loads((work/'selection.json').read_text()):
        ds=s['dataset']
        setting=json.loads((work/'selection'/ds/'binding.json').read_text())
        cfg=setting['source_config']
        model,binding=ev.restore(Path(setting['source_folder']),cfg,Path(cfg['data_root'])/'manifest.json',s['step'],'cpu')
        _,fixed,bank=assets(work.parent/'coefficient-20261116/assets',ds,'selection','cpu')
        entrances={}
        hooks=[layer.register_forward_pre_hook(lambda _,args,n=name:entrances.update({n:args[0].clone()}))
               for name,layer in model.ode_layers()]
        model(fixed['x'][:1024],fixed['t'][:1024])
        for h in hooks:
            h.remove()
        agg=torch.zeros(5,dtype=torch.double)
        for name,layer in model.ode_layers():
            y=entrances[name].double()
            p=layer.operator().double()
            dt=1/layer.substeps
            linear=torch.nn.functional.linear
            for _ in range(layer.substeps):
                d=layer.step_size(y)
                delta=0.5*dt*linear(y*d,p)
                half=y+delta
                native=linear(half*layer.step_size(half),p)
                stale=linear(half*d,p)
                correction=native-stale
                expected=linear(half*(layer.step_size(half)-d),p)
                torch.testing.assert_close(correction,expected,atol=1e-12,rtol=1e-10)
                predicted=linear((d.pow(3)-d)*delta,p)
                agg+=torch.stack([correction.square().sum(),stale.square().sum(),
                                  predicted.square().sum(),(predicted*correction).sum(),
                                  (predicted-correction).square().sum()])
                y=y+dt*native
            reference=layer(entrances[name]).double()
            torch.testing.assert_close(y,reference,atol=5e-5,rtol=2e-5)
        c,st,pred,dot,res=agg.tolist()
        result.append(dict(dataset=ds,queries=1024,layers=len(entrances),steps_per_layer=4,
                           correction_over_stale_drift_rms=(c/st)**.5,
                           linear_prediction_cosine=dot/(c*pred)**.5,
                           relative_linear_residual_rms=(res/c)**.5,
                           step=s['step'],assets_sha256=bank['sha256'],**binding))
    base.dump(out,result)

if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
