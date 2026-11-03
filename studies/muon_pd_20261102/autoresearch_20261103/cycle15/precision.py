"""Follow-up checkpoint precision and component counts; saved arrays only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from analyze import sha,write,HERE,STUDY,TAG


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--parent-runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();runtime=Path(a.runtime);parent=Path(a.parent_runtime);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    grid=json.loads((HERE/'grid.json').read_text());data={};bank_pairs=[];parent_case_cache={}
    for cell in grid['cells']:
        folder=parent/'lenovo/output/professor_validation14_20261104'/cell['dataset']/cell['source_case']/'seed000'
        records=[json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
        selected=sorted([r for r in records if r['checkpoint_step']==cell['checkpoint_step'] and r['heun_steps']==128],key=lambda r:r['bank'])
        assert [r['bank'] for r in selected]==list(range(8))
        with np.load(folder/'banks.npz',allow_pickle=False) as f:banks={k:f[k].copy() for k in f.files}
        generated=[]
        for row in selected:
            file=folder/row['generated_file'];assert sha(file)==row['generated_sha256'];generated.append(np.load(file,allow_pickle=False))
        data[cell['id']]=generated;parent_case_cache[cell['source_case']]=(cell['family'],records)
    for family,records in parent_case_cache.values():
        for bank in range(8):
            scores={step:next(r['sw2'] for r in records if r['checkpoint_step']==step and r['bank']==bank and r['heun_steps']==128) for step in [15000,20000]}
            bank_pairs.append(dict(family=family,bank=bank,sw2_15k=scores[15000],sw2_20k=scores[20000],delta_20k_minus15k=scores[20000]-scores[15000]))
    write(out/'bank-checkpoint-differences.csv',bank_pairs)
    direction=banks['directions'].astype(float);rows=[]
    for omitted in [-1]+list(range(8)):
        keep=[b for b in range(8) if b!=omitted];target=np.concatenate([banks[f'target_{b}'] for b in keep]);target2=np.concatenate([banks[f'target2_{b}'] for b in keep])
        target_sorted=np.sort(target.astype(float)@direction.T,axis=0);floor=float(np.sqrt(np.mean((target_sorted-np.sort(target2.astype(float)@direction.T,axis=0))**2)))
        for cell in grid['cells']:
            values=np.concatenate([data[cell['id']][b] for b in keep]);score=float(np.sqrt(np.mean((np.sort(values.astype(float)@direction.T,axis=0)-target_sorted)**2)))
            rows.append(dict(cell=cell['id'],family=cell['family'],checkpoint_step=cell['checkpoint_step'],omitted_bank=omitted,samples=len(values),sw2=score,floor=floor))
        print(f'Pooled omitted={omitted}: {len(rows)}/54',flush=True)
    write(out/'checkpoint-pools.csv',rows)
    spec=importlib.util.spec_from_file_location('transport_precision_native_data',STUDY/'reference/synthetic/data.py')
    native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
    manifest=json.loads((runtime/'audit/manifest.json').read_text());law=native.GaussianMixture(**manifest['datasets']['gmm_ring8_2d']['distribution'])
    mode_rows=[];count_rows=[]
    def modes(label,x,target_freq):
        posterior=law._terms(x,np.ones((len(x),1)))[3];assigned=posterior.argmax(axis=1);counts=np.bincount(assigned,minlength=len(law.weights));freq=counts/len(x)
        residual=x.astype(float)-law.means[assigned]
        mode_rows.append(dict(label=label,samples=len(x),l1_to_analytic_weights=float(np.abs(freq-law.weights).sum()),l1_to_empirical_target=float(np.abs(freq-target_freq).sum()),center_coordinate_mse=float(np.mean(residual**2)),assignment_confidence_min=float(posterior.max(axis=1).min()),low_confidence_fraction=float((posterior.max(axis=1)<.99).mean())))
        for component,n in enumerate(counts):count_rows.append(dict(label=label,component=component,count=int(n),frequency=float(freq[component]),analytic_weight=float(law.weights[component]),target_frequency=float(target_freq[component])))
    def target_freq(x):return np.bincount(law._terms(x,np.ones((len(x),1)))[3].argmax(axis=1),minlength=len(law.weights))/len(x)
    bank_target=banks['target_0'];freq0=target_freq(bank_target);modes('bank0_target',bank_target,freq0)
    for cell in grid['cells']:
        folder=runtime/'output'/TAG/cell['dataset']/cell['id']/'seed000';result=json.loads((folder/'result.json').read_text())
        for row in result['switches']:modes(f"{cell['family']}_{cell['checkpoint_step']}_{row['mode']}_{row['cut_step']}",np.load(folder/row['file'],allow_pickle=False),freq0)
    pooled_target=np.concatenate([banks[f'target_{b}'] for b in range(8)]);freq_pool=target_freq(pooled_target);modes('pool_target',pooled_target,freq_pool)
    for cell in grid['cells']:modes(f"pool_{cell['family']}_{cell['checkpoint_step']}",np.concatenate(data[cell['id']]),freq_pool)
    write(out/'component-summary.csv',mode_rows);write(out/'component-counts.csv',count_rows)
    receipt=dict(passed=True,new_model_calls=0,new_training_runs=0,kind='post_outcome_precision_and_component_diagnostic',pooled_rows=len(rows),bank_differences=len(bank_pairs),component_summaries=len(mode_rows),producer_sha256=sha(__file__),source_grid_sha256=sha(HERE/'grid.json'),primary_estimator_unchanged=True,all_models_checkpoints_and_omissions_retained=True)
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,(family,label) in zip(axes,[('pd_isru_k1','ISRU-D'),('free_isru','Free ISRU'),('swiglu64','Selected Swi64')]):
        for omitted in range(8):
            r=sorted([r for r in rows if r['family']==family and r['omitted_bank']==omitted],key=lambda r:r['checkpoint_step']);ax.plot([15,20],[x['sw2'] for x in r],color='gray',alpha=.35)
        full=sorted([r for r in rows if r['family']==family and r['omitted_bank']==-1],key=lambda r:r['checkpoint_step']);ax.plot([15,20],[x['sw2'] for x in full],'o-',color='#0072B2',label='All8 banks pooled')
        ax.set(title=label,xlabel='Checkpoint (thousand updates)',ylabel='Pooled SW2',xticks=[15,20]);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7)
    fig.savefig(out/'checkpoint-precision.png',dpi=180);plt.close(fig)
    print(json.dumps(receipt))


if __name__=='__main__':main()
