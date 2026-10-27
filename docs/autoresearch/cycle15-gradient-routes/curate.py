"""Curate only independently audited frozen gradient probes."""
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
root=args.root;here=Path(__file__).resolve().parent;out=here/'results';out.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text())
assert load(root/'audit-v2/audit.json')['status']=='PASS'
receipt=load(root/'output/receipt.json');assert receipt['status']=='PASS'
for src,dst in [('output/gradient-groups.csv','gradient-groups.csv'),('output/checkpoints.csv','checkpoints.csv'),
    ('output/receipt.json','receipt.json'),('audit-v2/audit.json','audit.json'),('preflight.json','preflight.json')]:
    shutil.copyfile(root/src,out/dst)
g=list(csv.DictReader((out/'gradient-groups.csv').open()));c=list(csv.DictReader((out/'checkpoints.csv').open()))
group={(r['law'],r['family'],int(r['step']),r['group']):r for r in g}
check={(r['law'],r['family'],int(r['step'])):r for r in c}
families=list(dict.fromkeys(r['family'] for r in c));final=[]
for law in ('thin','thick'):
    for family in families:
        r=check[law,family,5000];h=group[law,family,5000,'hidden'];p=group[law,family,5000,'P']
        loss=sum(float(r[k]) for k in ('tangent_loss','forced_loss','remaining_normal_loss'))
        final.append(dict(law=law,family=family,forced_loss_fraction=float(r['forced_loss'])/loss,
            native_clip=float(r['native_clip']),full_clip=float(h['full_clip']),removed_clip=float(h['forced_removed_clip']),
            hidden_update_change=float(h['update_relative_change']),hidden_update_cosine=float(h['update_cosine']),
            P_cosine=float(p['tangent_remaining_cosine']) if p['tangent_remaining_cosine'] else None,
            P_cancellation=float(p['cancellation_ratio']) if p['cancellation_ratio'] else None))
with (out/'final-summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(final[0]));w.writeheader();w.writerows(final)
figdir=out/'figures';figdir.mkdir();plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
x=np.arange(len(families));fig,axes=plt.subplots(2,3,figsize=(14,6.3),sharex=True)
for i,law in enumerate(('thin','thick')):
    rows=[r for r in final if r['law']==law]
    ax=axes[i,0];ax.bar(x,[100*r['forced_loss_fraction'] for r in rows],color='#4c78a8')
    ax.set_ylabel(f'{law}: forced loss (%)');ax.set_ylim(0,100)
    ax=axes[i,1];ax.bar(x,[100*float(group[law,family,1000,'hidden']['update_relative_change']) for family in families],color='#e28b36')
    ax.set_ylabel(f'{law}: step1000 update change (%)');ax.set_ylim(0,26)
    ax=axes[i,2];ax.bar(x,[100*r['hidden_update_change'] if r['hidden_update_change']>1e-12 else 0 for r in rows],color='#e28b36')
    ax.set_ylabel(f'{law}: step5000 update change (%)');ax.set_ylim(0,26)
for ax in axes[-1]:ax.set_xticks(x,families,rotation=45,ha='right')
fig.suptitle('Native next-batch probes: final loss energy and local Adam update effects')
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(figdir/f'loss-versus-update.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,6.3),sharex=True)
for i,law in enumerate(('thin','thick')):
    for family in families:
        steps=[1000,2500,5000];rows=[group[law,family,k,'hidden'] for k in steps]
        axes[i,0].plot(steps,[float(r['forced_removed_clip'])/float(r['full_clip']) for r in rows],label=family,marker='.',lw=1)
        pr=[group[law,family,k,'P'] for k in steps]
        if pr[0]['tangent_remaining_cosine']:
            axes[i,1].plot(steps,[float(r['tangent_remaining_cosine']) for r in pr],label=family,marker='.',lw=1)
    axes[i,0].axhline(1,color='black',lw=.6);axes[i,1].axhline(0,color='black',lw=.6)
    axes[i,0].set_ylabel(f'{law}: clip removed/full');axes[i,1].set_ylabel(f'{law}: P tangent/normal cosine')
    axes[i,1].set_ylim(-1.05,1.05)
for ax in axes[-1]:ax.set_xlabel('Saved checkpoint step')
axes[0,0].legend(ncol=3,fontsize=6,loc='upper center');fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(figdir/f'clip-and-P-directions.{ext}',dpi=170)
plt.close(fig)
# Descriptive ranges only: these models and checkpoints are not independent repeats.
summary={}
for law in ('thin','thick'):
    rows=[r for r in final if r['law']==law]
    summary[law]={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in
        ('forced_loss_fraction','hidden_update_change','hidden_update_cosine','full_clip','removed_clip')}
for step in (1000,2500):
    rows=[r for r in g if r['law']=='thick' and int(r['step'])==step and r['group']=='hidden']
    summary[f'thick_step{step}']={k:[min(float(r[k]) for r in rows),max(float(r[k]) for r in rows)] for k in ('update_relative_change','full_clip','forced_removed_clip','update_cosine')}
summary['max_hidden_forced_norm']=max(float(r['forced_norm']) for r in g if r['group']=='hidden')
summary['native_promoted_gradient_norm_max_abs']=max(abs(float(r['native_gradient_norm'])-float(r['promoted_gradient_norm'])) for r in c)
summary['decision']='Diagnostic only; no loss/head/optimizer rescue and no architecture nomination.'
(out/'decision.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'results.tsv').open('w') as f:
    f.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    f.write(f"{load(root/'settings.json')['registration_commit'][:7]}\t{summary['max_hidden_forced_norm']:.12g}\t{receipt['peak_rss_GiB']:.3f}\tkeep\tFrozen gradient diagnostic only; no architecture kept\n")
header='| Law | Family | Forced loss % | Clip full / removed | Hidden Adam change % | P tangent/normal cosine |\n|---|---|---:|---:|---:|---:|\n'
rows=''.join(f"| {r['law']} | {r['family']} | {100*r['forced_loss_fraction']:.2f} | {r['full_clip']:.4f} / {r['removed_clip']:.4f} | {100*r['hidden_update_change']:.3f} | {r['P_cosine']:.3f} |\n" if r['P_cosine'] is not None else f"| {r['law']} | {r['family']} | {100*r['forced_loss_fraction']:.2f} | {r['full_clip']:.4f} / {r['removed_clip']:.4f} | {100*r['hidden_update_change']:.3f} | — |\n" for r in final)
(here/'numeric-table.md').write_text('# Final native next-batch probes\n\n'+header+rows+'\nAdam columns are promoted one-step predictions with saved moments, not actual retraining.\n')
print(json.dumps(summary,indent=2))
