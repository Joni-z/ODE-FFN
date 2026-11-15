"""Frozen-K8 term measurements and full-model truncation interventions."""
import argparse
import json
from pathlib import Path
import numpy as np
from order_trial import HERE,base,torch
from evaluate_order import parent as ev


def rms(x):return float(x.double().square().mean().sqrt())


@torch.no_grad()
def term_metrics(model,x,t):
    captured={};hooks=[]
    for name,layer in model.ode_layers():
        def record(module,args,output,key=name):captured[key]=(args[0].detach().clone(),output.detach().clone())
        hooks.append(layer.register_forward_hook(record))
    model(x,t)
    for hook in hooks:hook.remove()
    rows=[];exponential=[]
    for name,layer in model.ode_layers():
        h,y=captured[name];d=layer.step_size(h);p=layer.operator();term=h;terms=[];total=h
        for order in range(1,9):
            term=torch.nn.functional.linear(term*d,p)/order;terms.append(term);total=total+term
        assert torch.equal(total,y),'Measurement must reproduce the native recurrence exactly'
        h_rms=max(rms(h),1e-30);first=max(rms(terms[0]),1e-30)
        higher=sum(terms[1:]);increment=sum(terms)
        rows.append(dict(layer=name,input_rms=h_rms,term_rms_over_input=[rms(v)/h_rms for v in terms],higher_rms_over_first=rms(higher)/first,increment_rms_over_input=rms(increment)/h_rms,output_rms_over_input=rms(total)/h_rms,cancellation_ratio=sum(rms(v) for v in terms)/max(rms(increment),1e-30),entrances=len(h)))
        for i in range(8):
            a=(p.double()*d[i].double()[None,:]).cpu();u=h[i].double().cpu()
            v=u;polynomial=u
            for j in range(1,9):v=a@v/j;polynomial=polynomial+v
            exact=torch.matrix_exp(a)@u
            relative=float(torch.linalg.vector_norm(polynomial-exact)/torch.linalg.vector_norm(exact).clamp_min(1e-30))
            exponential.append(dict(layer=name,entrance=i,relative_exp_error=relative if np.isfinite(relative) else None,finite=bool(torch.isfinite(exact).all() and torch.isfinite(polynomial).all()),generator_sigma_max=float(torch.linalg.matrix_norm(a,ord=2))))
    return rows,exponential


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--witness',action='store_true');p.add_argument('--step',type=int,default=20000)
    a=p.parse_args();torch.set_num_threads(1);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    folder=Path(a.folder);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell'];cache=base.verify_data(a.data_root,cell);manifest=cache/'manifest.json'
    spec=json.loads(manifest.read_text())['datasets'][cell['dataset']];law=base.train.load_distribution(spec)
    fixed=base.train.fixed_eval_data(cache/cell['dataset'],law,law.dim,0,torch.device(a.device),4096,256,'test')
    model,identity=ev.restore(folder,cfg,manifest,a.step,a.device);snapshot=ev.tensor_snapshot(model)
    assert all(layer.orders==8 for _,layer in model.ode_layers())
    terms,exponential=term_metrics(model,fixed['x'][:512],fixed['t'][:512])
    base.dump(out/'terms.json',dict(layers=terms,exponential=exponential,**identity))
    evaluation=Path(a.evaluation)
    if a.witness:
        arrays=ev.bank_arrays(law,fixed['directions'].cpu().numpy(),64,1,2026110400)
    else:
        assert json.loads((evaluation/'COMPLETE.json').read_text())['passed']
        with np.load(evaluation/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
        hashes={k:ev.array_hash(v) for k,v in arrays.items()}
        assert hashes==json.loads((evaluation/'binding.json').read_text())['bank_hashes']
        reference=next(r for r in json.loads((evaluation/'summary.json').read_text()) if r['step']==a.step and r['variant']=='native')
        assert reference['checkpoint_sha256']==identity['checkpoint_sha256']
    # K8 identity test is small; the full K8 samples have already been evaluated.
    source=torch.from_numpy(arrays['source_0']).to(a.device)
    if not a.witness:
        anchor=base.train.generated_samples(model,source,128).cpu().numpy()
        assert np.array_equal(anchor,np.load(evaluation/f'{a.step}-native-b0.npy'))
    rows=[];banks=[]
    for order in (1,2,4):
        for _,layer in model.ode_layers():layer.orders=order
        _,mse=ev.field(model,fixed)
        eq=ev.equivariance(model,fixed['x'][:1024],fixed['t'][:1024]) if law.dim==2 else None
        values=[];targets=[];scores=[]
        for bank in range(1 if a.witness else 8):
            source=torch.from_numpy(arrays[f'source_{bank}']).to(a.device)
            target=torch.from_numpy(arrays[f'target_{bank}']).to(a.device)
            generated=base.train.generated_samples(model,source,128)
            assert bool(torch.isfinite(generated).all())
            value=generated.cpu().numpy();values.append(value);targets.append(target.cpu().numpy())
            score=base.train.sliced_w2(generated.double(),target.double(),fixed['directions'].double());scores.append(score)
            if not a.witness:np.save(out/f'order{order}-b{bank}.npy',value,allow_pickle=False)
            banks.append(dict(order=order,bank=bank,sw2=score))
        values=np.concatenate(values);targets=np.concatenate(targets)
        pool=base.train.sliced_w2(torch.from_numpy(values).to(a.device).double(),torch.from_numpy(targets).to(a.device).double(),fixed['directions'].double())
        row=dict(order=order,field_mse=mse,sw2_mean=float(np.mean(scores)),pooled_sw2=pool,equivariance=eq)
        if law.dim==2:row.update(ev.mode_stats(values,law))
        rows.append(row);base.dump(out/'truncation.json',rows);base.dump(out/'banks.json',banks)
    for _,layer in model.ode_layers():layer.orders=8
    ev.same_tensors(model,snapshot)
    base.dump(out/'COMPLETE.json',dict(passed=True,witness=a.witness,checkpoint=identity,source_sha256=base.digest(__file__),all_parameters_buffers_unchanged=True,orders_restored=8))
    print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
