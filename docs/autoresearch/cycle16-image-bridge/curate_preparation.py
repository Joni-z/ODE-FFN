"""Plot data/calibration evidence; this produces no model-quality result."""
from pathlib import Path
import argparse,csv,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
here=Path(__file__).resolve().parent;out=here/'results';out.mkdir(exist_ok=False);fig=out/'figures';fig.mkdir()
load=lambda p:json.loads(p.read_text())
assert load(args.root/'preparation-audit.json')['status']=='PASS'
p=load(args.root/'preflight.json');rows=p['rows']
for name in ('preflight.json','semantics.json','preparation-audit.json'):shutil.copyfile(args.root/name,out/name)
shutil.copyfile(args.data/'manifest.json',out/'data-manifest.json')
with (out/'calibration.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
families=[r['family'] for r in rows if r['device']=='cpu'];x=np.arange(len(families))
f,ax=plt.subplots(figsize=(7,3.5));ax.spines[['top','right']].set_visible(False)
for offset,device,color in ((-.18,'cpu','#4c78a8'),(.18,'mps','#e28b36')):
    a=[r for r in rows if r['device']==device];bars=ax.bar(x+offset,[1000*r['median_seconds_per_update'] for r in a],width=.36,label=device.upper(),color=color)
    ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
ax.set_xticks(x,families);ax.set_ylabel('Median milliseconds per update');ax.set_ylim(0,125)
ax.set_title('Mac mini: batch64, existing JiT image backbone');ax.legend();f.tight_layout()
for ext in ('png','pdf'):f.savefig(fig/f'device-calibration.{ext}',dpi=160)
plt.close(f)
images=np.load(args.data/'images.uint8.npy',mmap_mode='r')
with np.load(args.data/'split.npz') as d:idx=np.concatenate((d['train'][:32],d['validation'][:32]))
f,axes=plt.subplots(8,8,figsize=(6,6.4))
for ax,i in zip(axes.flat,idx):ax.imshow(images[i].transpose(1,2,0));ax.axis('off')
f.suptitle('Real CIFAR-10 examples (not generated)\nRows1-4: training; rows5-8: validation',fontsize=11)
f.tight_layout(rect=(0,0,1,.94),pad=.15);f.savefig(fig/'real-data-examples.png',dpi=160);plt.close(f)
print(json.dumps(dict(status='CURATED',calibration_rows=len(rows),figures=2,quality_results=False)))
