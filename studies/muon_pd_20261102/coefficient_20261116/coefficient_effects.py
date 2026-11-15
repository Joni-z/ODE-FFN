"""Check whether learned coefficients change actual features, not just scalars."""
import argparse,copy,json
from pathlib import Path
from coefficient_trial import base,torch
import coefficient_evaluate as evaluation

def mapping(layer,h,d,duration):
    term=h;total=h;p=layer.operator()
    for j in range(1,9):
        term=duration*torch.nn.functional.linear(term*d,p)/j
        total=total+(term if j==1 else layer.alpha[j-2]*term)
    return total

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);assert not out.exists();torch.set_num_threads(1);rows=[]
    for case in json.loads((root/'anchors.json').read_text())['cases']:
        if case['existing']:continue
        folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cache=base.verify_data(a.data_root,cfg['cell'])
        model,identity=evaluation.evaluator.ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu')
        arrays,fixed,binding=evaluation.evaluator.assets(root/'assets',case['dataset'],'selection','cpu')
        snapshot=evaluation.evaluator.ev.tensor_snapshot(model);captures={};hooks=[]
        for name,layer in model.ode_layers():
            def hook(m,args,key=name):captures[key]=args[0][:16].detach().clone()
            hooks.append(layer.register_forward_pre_hook(hook))
        with torch.no_grad():model(fixed['x'][:16],fixed['t'][:16])
        for hook in hooks:hook.remove()
        prediction,mse=evaluation.evaluator.ev.field(model,fixed);reset=copy.deepcopy(model)
        with torch.no_grad():
            for _,layer in reset.ode_layers():layer.alpha.fill_(1)
        reset_prediction,reset_mse=evaluation.evaluator.ev.field(reset,fixed);layers=[]
        with torch.no_grad():
            for name,layer in model.ode_layers():
                h=captures[name];d=layer.step_size(h);full=mapping(layer,h,d,1.)
                assert torch.allclose(full,layer(h),atol=1e-6,rtol=1e-6)
                half=mapping(layer,h,d,.5);same=mapping(layer,half,d,.5);restart=mapping(layer,half,layer.step_size(half),.5)
                denom=(full-h).square().mean().sqrt().clamp_min(1e-12)
                layers.append(dict(layer=name,alpha=layer.alpha.detach().tolist(),
                    same_generator_defect_relative=float((same-full).square().mean().sqrt()/denom),
                    restarted_defect_relative=float((restart-full).square().mean().sqrt()/denom)))
        evaluation.evaluator.ev.same_tensors(model,snapshot)
        rows.append(dict(id=case['id'],checkpoint=identity,assets_sha256=binding['sha256'],original_field_mse=mse,
             coefficients_reset_field_mse=reset_mse,prediction_shift_rms=float(((prediction-reset_prediction)**2).mean()**.5),layers=layers))
    base.dump(out,dict(passed=True,scope='frozen-weight intervention, not retraining comparison',rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
