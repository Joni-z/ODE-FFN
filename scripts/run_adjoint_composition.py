"""Frozen complete-field decomposition by residual derivative path length."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.adjoint_composition import AdjointComposition
from synthetic.conservative_readout import build
from synthetic.train import restore_model

DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
FAMILIES=('ode_k2','swiglu128','swiglu192')
TIMES=(.1,.3,.5,.7,.9)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def component_fields(model,x,t):
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),512):
            out.append(model.components(torch.from_numpy(x[i:i+512]).double(),torch.from_numpy(t[i:i+512]).double()).numpy())
    return np.concatenate(out)


def component_jacobians(model,x,t):
    values=[];derivatives=[]
    for start in range(0,len(x),128):
        xx=torch.from_numpy(x[start:start+128]).double().requires_grad_(True)
        tt=torch.from_numpy(t[start:start+128]).double()
        components=model.components(xx,tt)
        rows=[]
        for degree in range(4):
            jac=[]
            for coordinate in range(x.shape[1]):
                last=degree==3 and coordinate==x.shape[1]-1
                jac.append(torch.autograd.grad(components[:,degree,coordinate].sum(),xx,retain_graph=not last)[0])
            rows.append(torch.stack(jac,1))
        values.append(components.detach().numpy());derivatives.append(torch.stack(rows,1).detach().numpy())
    return np.concatenate(values),np.concatenate(derivatives)


def main(parent,orientation,canonical,output):
    torch.set_num_threads(1)
    manifest=load(ROOT/'source-sha256.json')
    for name,value in manifest.items():assert sha(ROOT/name)==value
    assert load(canonical/'audit.json')['status']=='PASS'
    canonical_manifest=load(canonical/'report-file-sha256.json')
    for name,value in canonical_manifest.items():assert sha(canonical.parent/name)==value
    output.mkdir(parents=True,exist_ok=False)
    bindings={};risk=[];energies=[];jac_rows=[];skew_rows=[];finite=[];gradient_rows=[];grams=[]
    process=psutil.Process();started=time.monotonic();telemetry=[]
    max_value=max_jac=max_fd=max_gradient=0.
    def progress(**kwargs):
        row=dict(pid=process.pid,elapsed_seconds=time.monotonic()-started,rss_MiB=process.memory_info().rss/2**20,
            available_GiB=psutil.virtual_memory().available/2**30,**kwargs)
        telemetry.append(row)
        write(output/'status.json',dict(state='RUNNING',**row))
        with (output/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    for dataset in DATASETS:
        bank=orientation/'output/banks'/dataset
        path=bank/'field.npz';bindings[str(path)]=sha(path)
        with np.load(path) as data:x,t,truth=[data[key] for key in ('x','t','oracle')]
        for family in FAMILIES:
            for trained,runtime in (('gradient',parent/'output'),('directional',orientation/'output')):
                progress(dataset=dataset,family=family,trained=trained,phase='complete_fields')
                path=runtime/'runs'/dataset/(family+'_'+trained)/'step_0005000.pt';bindings[str(path)]=sha(path)
                state=torch.load(path,map_location='cpu',weights_only=False)
                reference=build(x.shape[1],family,'gradient').double().eval();restore_model(reference,state)
                model=AdjointComposition(reference.raw).eval()
                for parameter in model.parameters():parameter.requires_grad_(False)
                folder=output/'cases'/dataset/family/trained;folder.mkdir(parents=True)
                components=component_fields(model,x,t);partial=np.cumsum(components,axis=1);full=partial[:,-1]
                path=canonical/'crossovers'/dataset/family/trained/'fields.npz';bindings[str(path)]=sha(path)
                with np.load(path) as data:expected=data['prediction_A']
                np.testing.assert_allclose(full,expected,rtol=1e-9,atol=1e-10)
                max_value=max(max_value,float(np.max(np.abs(full-expected))))
                errors=np.mean((partial-truth[:,None,:])**2,axis=-1)
                delta=partial-full[:,None,:]
                alignment=2*np.mean((full-truth)[:,None,:]*delta,axis=-1)
                change_energy=np.mean(delta**2,axis=-1)
                np.testing.assert_allclose(errors-errors[:,-1,None],alignment+change_energy,rtol=1e-9,atol=1e-10)
                gram=np.einsum('nkd,nld->kl',components,components)/(len(x)*x.shape[1])
                np.testing.assert_allclose(gram.sum(),np.mean(full**2),rtol=1e-10,atol=1e-11)
                grams.append(dict(dataset=dataset,family=family,trained=trained,kind='field',time=None,gram=gram.tolist()))
                np.savez_compressed(folder/'field.npz',components=components,partial=partial,errors=errors,
                    alignment=alignment,change_energy=change_energy,component_gram=gram)
                for degree in range(4):
                    risk.append(dict(dataset=dataset,family=family,trained=trained,maximum_degree=degree,
                        oracle_mse=float(errors[:,degree].mean()),full_oracle_mse=float(errors[:,-1].mean()),
                        alignment=float(alignment[:,degree].mean()),change_energy=float(change_energy[:,degree].mean())))
                    energies.append(dict(dataset=dataset,family=family,trained=trained,degree=degree,
                        mean_squared_component=float(gram[degree,degree]),mean_squared_full=float(gram.sum())))
                for clock in TIMES:
                    path=bank/f'panel_{clock:.1f}.npz';bindings[str(path)]=sha(path)
                    with np.load(path) as data:px,pt,oracle_jac=[data[key] for key in ('x','t','oracle_jacobian')]
                    cv,cj=component_jacobians(model,px,pt)
                    partial_v=np.cumsum(cv,axis=1);partial_j=np.cumsum(cj,axis=1)
                    path=canonical/'crossovers'/dataset/family/trained/f'gradient-panel-{clock:.1f}.npz';bindings[str(path)]=sha(path)
                    with np.load(path) as data:ev,ej=data['prediction'],data['jacobian']
                    np.testing.assert_allclose(partial_v[:,-1],ev,rtol=1e-9,atol=1e-10)
                    np.testing.assert_allclose(partial_j[:,-1],ej,rtol=1e-8,atol=1e-9)
                    max_jac=max(max_jac,float(np.max(np.abs(partial_j[:,-1]-ej))))
                    skew=.5*(cj-cj.transpose(0,1,3,2))
                    gram=np.einsum('nkij,nlij->kl',skew,skew)/len(px)
                    full_skew=skew.sum(1)
                    np.testing.assert_allclose(gram.sum(),np.mean(np.sum(full_skew**2,axis=(-1,-2))),rtol=1e-7,atol=1e-10)
                    grams.append(dict(dataset=dataset,family=family,trained=trained,kind='skew',time=clock,gram=gram.tolist()))
                    np.savez_compressed(folder/f'panel_{clock:.1f}.npz',components=cv,component_jacobians=cj,skew_gram=gram)
                    for degree in range(4):
                        partial_skew=.5*(partial_j[:,degree]-partial_j[:,degree].transpose(0,2,1))
                        jac_rows.append(dict(dataset=dataset,family=family,trained=trained,time=clock,maximum_degree=degree,
                            jacobian_error=float(np.mean(np.sum((partial_j[:,degree]-oracle_jac)**2,axis=(-1,-2)))),
                            skew_energy=float(np.mean(np.sum(partial_skew**2,axis=(-1,-2))))))
                        skew_rows.append(dict(dataset=dataset,family=family,trained=trained,time=clock,degree=degree,
                            component_skew_energy=float(gram[degree,degree]),
                            full_skew_energy=float(np.mean(np.sum(full_skew**2,axis=(-1,-2)))),
                            component_energy_sum=float(np.trace(gram))))
                    xx=torch.from_numpy(px[:1]).double();tt=torch.from_numpy(pt[:1]).double()
                    direction=torch.cos(torch.arange(x.shape[1],dtype=torch.float64)).reshape(1,-1);direction/=direction.norm()
                    eps=1e-5
                    with torch.no_grad():
                        plus=model.components(xx+eps*direction,tt).cumsum(1)
                        minus=model.components(xx-eps*direction,tt).cumsum(1)
                    fd=((plus-minus)/(2*eps)).numpy()[0]
                    analytic=np.einsum('kij,j->ki',partial_j[0],direction.numpy()[0])
                    for degree in range(4):
                        error=float(np.linalg.norm(fd[degree]-analytic[degree])/max(1.,np.linalg.norm(analytic[degree])))
                        assert error<1e-5,(dataset,family,trained,clock,degree,error)
                        max_fd=max(max_fd,error)
                        finite.append(dict(dataset=dataset,family=family,trained=trained,time=clock,maximum_degree=degree,relative_error=error))
                for parameter in model.parameters():parameter.requires_grad_(True)
                xx=torch.from_numpy(x[:3]).double().requires_grad_(True);tt=torch.from_numpy(t[:3]).double().requires_grad_(True)
                args=(xx,tt,*model.parameters())
                av=model(xx,tt);bv=reference(xx,tt)
                ga=torch.autograd.grad(av.square().mean(),args)
                gb=torch.autograd.grad(bv.square().mean(),args)
                error=0.
                for a,b in zip(ga,gb):
                    torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-9)
                    error=max(error,float((a-b).abs().max()))
                max_gradient=max(max_gradient,error)
                gradient_rows.append(dict(dataset=dataset,family=family,trained=trained,maximum_absolute_gradient_error=error))
                for parameter in model.parameters():parameter.requires_grad_(False)
    table(output/'partial-field-errors.csv',risk);table(output/'component-energies.csv',energies)
    table(output/'partial-jacobian-errors.csv',jac_rows);table(output/'component-skew.csv',skew_rows)
    table(output/'finite-differences.csv',finite);table(output/'complete-gradient-checks.csv',gradient_rows)
    write(output/'component-grams.json',grams);write(output/'input-bindings.json',bindings)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    case_labels=[f'{family} / {trained}' for family in FAMILIES for trained in ('gradient','directional')]
    for ax,dataset,title in zip(axes,DATASETS,('Gaussian2D','Ring8','Product16')):
        matrix=np.array([[next(r['oracle_mse']/r['full_oracle_mse'] for r in risk if
            (r['dataset'],r['family'],r['trained'],r['maximum_degree'])==(dataset,family,trained,k)) for k in range(4)]
            for family in FAMILIES for trained in ('gradient','directional')])
        picture=ax.imshow(np.log10(matrix),aspect='auto',cmap='coolwarm',vmin=-1,vmax=2)
        for i in range(6):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
        ax.set(title=title,xticks=range(4),xticklabels=['0','0–1','0–2','0–3'],yticks=range(6),yticklabels=case_labels)
        ax.set_xlabel('Included residual-factor degrees')
    fig.colorbar(picture,ax=axes.ravel().tolist(),label='log10(partial/full field MSE)',shrink=.8)
    fig.subplots_adjust(left=.16,right=.86,wspace=.85,bottom=.16,top=.89)
    for suffix in ('png','pdf'):fig.savefig(output/f'partial-field-errors.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    for name,value in manifest.items():assert sha(ROOT/name)==value
    for name,value in bindings.items():assert sha(name)==value
    audit=dict(status='PASS',actual_learned_states=len(gradient_rows),complete_fields=len(gradient_rows),
        complete_panels=len(jac_rows)//4,all_partial_fields=len(risk),all_partial_panels=len(jac_rows),
        finite_differences=len(finite),maximum_full_value_error=max_value,maximum_full_J_error=max_jac,
        maximum_partial_fd_error=max_fd,maximum_complete_gradient_error=max_gradient,
        source_manifest_sha256=sha(ROOT/'source-sha256.json'),sampled_RSS_max_MiB=max(r['rss_MiB'] for r in telemetry),
        minimum_available_GiB=min(r['available_GiB'] for r in telemetry))
    write(output/'audit.json',audit);write(output/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-started))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--orientation',type=Path,required=True)
    p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.parent.resolve(),a.orientation.resolve(),a.canonical.resolve(),a.output.resolve())
