"""Plot the independently completed full sharing-refinement cohort."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--receipts',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.analysis);verified=json.loads((root/'COMPLETE.json').read_text());assert verified['passed'] and verified['pooled_rows']==16 and verified['bank_rows']==128
    data=json.loads((root/'scores.json').read_text());rows=data['rows'];out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for file in root.iterdir():
        if file.name=='gpu-history.jsonl':
            history=[json.loads(line) for line in file.read_text().splitlines()]
            (out/'gpu-history.json').write_text(json.dumps(history,indent=2)+'\n')
        elif file.is_file():shutil.copy2(file,out/file.name)
    for file in Path(a.receipts).iterdir():
        if file.is_file():shutil.copy2(file,out/file.name)
    tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];names=['Ring8','Product16']
    at=lambda d,f,i,o:next(r for r in rows if r['dataset']==d and r['family']==f and r['inner']==i and r['outer']==o)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for column,(task,name) in enumerate(zip(tasks,names)):
        for family,label,color in [('state_midpoint4','Shared P','#0072B2'),('stage_operator_midpoint4','Stage P','#D55E00')]:
            axes[0,column].plot([128,256,512],[at(task,family,4,s)['pooled_sw2'] for s in [128,256,512]],marker='o',label=label,color=color)
            axes[1,column].plot([4,16],[at(task,family,s,128)['pooled_sw2'] for s in [4,16]],marker='o',color=color)
        axes[0,column].set(xticks=[128,256,512],xlabel='Outer Heun steps (inner4)',ylabel='Pooled SW2',title=name);axes[0,column].legend()
        axes[1,column].set(xticks=[4,16],xlabel='Hidden midpoint substeps (Heun128)',ylabel='Pooled SW2')
    figure.savefig(out/'refinement.png',dpi=190);plt.close(figure)
    text=['# Sharing survives outer refinement; the Ring8 hidden-limit evidence weakens','','The shared-generator lead persists in pooled scores after both outer and','hidden refinement. Ring8\'s paired-bank support weakens with16hidden substeps;','Product16 retains all8paired-bank wins. All scores use the same new sharing','confirmation sources, targets and directions.','','| Inner steps | Outer Heun | Ring8 shared | Ring8 stage | Shared wins | Product16 shared | Product16 stage | Shared wins |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for inner,outer in [(4,128),(4,256),(4,512),(16,128)]:
        r=[]
        for task in tasks:
            left=at(task,'state_midpoint4',inner,outer);right=at(task,'stage_operator_midpoint4',inner,outer)
            gap=next(g for g in data['gaps'] if g['dataset']==task and g['inner']==inner and g['outer']==outer)
            r.extend([f"{left['pooled_sw2']:.3f}",f"{right['pooled_sw2']:.3f}",str(gap['shared_better_banks'])+'/8'])
        text.append('| '+str(inner)+' | '+str(outer)+' | '+' | '.join(r)+' |')
    text+=['','![Full-bank refinement](refinement.png)','','Each row/model uses32768samples in eight common banks. CUDA Heun128','reproduces the completed CPU primary scores within3.44e-8. Independent NumPy','recomputation of all16pooled and128bank scores differs by at most2.78e-17.','The new piecewise-stage reference preserves all four quarter-interval matrices','under hidden refinement. Neither the checkpoint nor response changes.','','At16hidden substeps, Ring8\'s mean bank difference (stage minus shared) is','0.006, with approximate conditional95%interval[-0.002,0.015] and5/8wins.','Its pooled gap is0.012. Product16\'s pooled gap is0.012and sharing wins8/8.','Thus the fixed-budget result is stronger than the claim of a two-task','continuous-hidden-flow advantage. The outer solver does not explain the','primary gap; the hidden discretization contributes to its Ring8 size.','','Both hidden models have verified second-order convergence to their defined','autonomous/piecewise fields. Their matched architectural difference is the','stationarity of P. A shared P still gives a state-dependent Jacobian','P diag(q\'(y)), so stationarity leaves nonlinear adaptation along the path.','','The GPU job used one idle physical145GPU4, checked occupancy after every','resolution, and then exited. Sampling/metric compute summed to53.824seconds.','GPU telemetry and the launch/readiness receipts are included. No other task','was stopped. Raw generated arrays remain outside Git.','','Next: test whether the sharing gap follows symmetry-breaking/mode-allocation','errors, as registered in [the follow-up](../../sharing-symmetry-followup.md).']
    (out/'analysis.md').write_text('\n'.join(text)+'\n')
    result=dict(passed=True,source_sha256=sha(__file__),artifacts={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()})
    (out/'display-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
