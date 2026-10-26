"""Plot paired supervision controls without treating the oracle as a candidate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(studies: Path, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    rows=[];locations={};hashes={}
    for study in sorted(studies.iterdir()):
        path=study/'audit/audit.json'
        audit=json.loads(path.read_text())
        assert audit['state']=='PASS' and audit['population_stream_regenerated']
        assert audit['population_stream_replay_batches']==5000
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for row in audit['rows']:rows.append(row);locations[row['dataset']]=study
    assert len(rows)==12
    datasets=['gmm_ring8_2d','rotated_product_bimodal_16d']
    families=['vector','swiglu_budget','swiglu']
    names=['Vector ODE','SwiGLU small','SwiGLU full']
    colors=['#2678b2','#638c43','#222222']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.5),layout='constrained')
    for ax,dataset,title in zip(axes,datasets,['Ring8','Rotated product16']):
        x=np.arange(3)
        for offset,supervision,color in [(-.18,'pair','#2678b2'),(.18,'mean','#db8321')]:
            values=[next(r['validation_oracle_mse'] for r in rows if r['dataset']==dataset and r['label']==family+'_'+supervision) for family in families]
            ax.bar(x+offset,values,width=.33,color=color,label='Pair target' if supervision=='pair' else 'Exact conditional mean')
        ax.set_xticks(x,names,rotation=15);ax.set_yscale('log');ax.set_title(title)
    axes[0].set_ylabel('Validation population-oracle MSE')
    axes[1].legend(frameon=False,fontsize=8)
    for suffix in ('pdf','png'):fig.savefig(output/f'population-noise-endpoints.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.0),layout='constrained',sharey='row')
    for row_index,dataset in enumerate(datasets):
        for column,supervision in enumerate(['pair','mean']):
            ax=axes[row_index,column];study=locations[dataset]
            for family,name,color in zip(families,names,colors):
                label=family+'_'+supervision
                cfg=json.loads((study/'configs'/f'{label}.json').read_text())
                kind=cfg['models'][0]
                path=study/'runs'/cfg['experiment_id']/dataset/kind/'seed000/metrics.jsonl'
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                curve=[json.loads(line) for line in path.read_text().splitlines()]
                ax.plot([v['step'] for v in curve],[v['validation_oracle_mse'] for v in curve],color=color,label=name)
            ax.set_yscale('log');ax.set_xlabel('Training update')
            ax.set_title(('Ring8' if row_index==0 else 'Product16')+' / '+('pair labels' if column==0 else 'mean labels'))
        axes[row_index,0].set_ylabel('Population-oracle MSE')
    axes[1,1].legend(frameon=False,fontsize=8)
    for suffix in ('pdf','png'):fig.savefig(output/f'population-noise-learning.{suffix}',dpi=180)
    plt.close(fig)
    (output.parent/'plot-provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=hashes,
        interpretation='Paired fixed-protocol diagnostic. No inferential error bars or oracle-dependent architecture claim.'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('studies','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();plot(args.studies,args.output)
