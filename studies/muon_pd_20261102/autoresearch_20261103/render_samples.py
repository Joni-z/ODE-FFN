"""Curate frozen sample readouts and plot the registered latent-coordinate cases."""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--cases',required=True);p.add_argument('--out',required=True);a=p.parse_args()
receipt=Path(a.receipt);cases=Path(a.cases);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
data=json.loads(receipt.read_text());assert len(data['rows'])==74 and len(data['histograms'])==21 and data['unique_generated_arrays']==71
assert hashlib.sha256(cases.read_bytes()).hexdigest()==data['cases_manifest_sha256']
assert hashlib.sha256(Path(__file__).with_name('sample_structure.py').read_bytes()).hexdigest()==data['analysis_source_sha256']
shutil.copyfile(receipt,out/'receipt.json');shutil.copyfile(cases,out/'cases.json')
array_keys=['axis_w2','axis_gap_mass','axis_near_mode_mse','axis_second_moment']
rows=[{k:v for k,v in row.items() if k not in array_keys} for row in data['rows']]
axes_rows=[dict(group=row['group'],cell=row['cell'],axis=j,**{key:row[key][j] for key in array_keys}) for row in data['rows'] for j in range(16)]
def save(name,values):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
save('endpoints.csv',rows);save('axis-readouts.csv',axes_rows)
hist=data['histograms'];labels={h['label']:(h['group'],h['cell']) for h in hist};lookup={(row['group'],row['cell']):row for row in data['rows']}
save('case-studies.csv',[dict(label=label,**{k:v for k,v in lookup[key].items() if k not in array_keys}) for label,key in labels.items()])
reference=data['reference'];locations=np.asarray(reference['locations']);variances=np.asarray(reference['variances'])
for name,order in [('ode-cases',['Scalar K2','Raw K1','Abs K1']),('ordinary-cases',['SwiGLU128','SwiGLU192','Constant ordinary','Bilinear64 AdamW'])]:
    fig,axes=plt.subplots(len(order),3,figsize=(10,2.1*len(order)),squeeze=False)
    for row,label in enumerate(order):
        for col,j in enumerate([0,7,15]):
            record=next(x for x in hist if x['label']==label and x['axis']==j);ax=axes[row,col]
            edges=np.asarray(record['edges']);density=np.asarray(record['counts'])/(record['sample_count']*np.diff(edges))
            ax.stairs(density,edges,color='#4477aa',label='Saved generated samples')
            points=np.linspace(edges[0],edges[-1],800);sigma=math.sqrt(variances[j]);mean=locations[j]
            target=.5/(sigma*math.sqrt(2*math.pi))*(np.exp(-.5*((points-mean)/sigma)**2)+np.exp(-.5*((points+mean)/sigma)**2))
            ax.plot(points,target,'k--',lw=1.2,label='Exact population marginal')
            max_density=max(max(np.asarray(x['counts'])/(x['sample_count']*np.diff(x['edges']))) for x in hist if x['axis']==j)
            ax.set_xlim(edges[0],edges[-1]);ax.set_ylim(0,max(max_density,float(target.max()))*1.08)
            metric=lookup[labels[label]];gap=metric['axis_gap_mass'][j];expected=reference['population_gap_mass'][j]
            ax.text(.03,.96,f'Gap mass {100*gap:.2f}% / target {100*expected:.2f}%',transform=ax.transAxes,va='top',fontsize=7)
            if col==0:ax.set_ylabel(label+'\nDensity',fontsize=9)
            if row==0:ax.set_title(f'Latent axis {j}, modes +/-{mean:.2f}',fontsize=10)
            if row==len(order)-1:ax.set_xlabel('Latent coordinate')
            ax.grid(alpha=.15)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('Product16: frozen Heun32 samples versus the known two-mode marginals',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5));seen=set()
for row in data['rows']:
    if row['generated_sha256'] in seen:continue
    seen.add(row['generated_sha256']);ax.scatter(row['primary_random_sw2'],row['latent_axis_sw2'],color='#999999',alpha=.6,s=22)
for i,(label,key) in enumerate(labels.items()):
    row=lookup[key];ax.scatter(row['primary_random_sw2'],row['latent_axis_sw2'],s=55,label=label)
ax.axhline(reference['target_axis_w2_reference'],c='gray',ls=':',label='Finite target-target axis reference')
ax.set(title='Frozen Product16 readouts: 71 distinct generated arrays',xlabel='Primary random-direction SW2',ylabel='Supplementary latent-axis SW2');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'direction-readouts.{ext}',dpi=180,bbox_inches='tight')
print(out)
