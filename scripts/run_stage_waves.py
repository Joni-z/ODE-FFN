"""Run one dataset's predeclared waves on one GPU, then audit all chosen checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthetic.models import VelocityNet
from synthetic.train import run, fixed_eval_data, generated_samples, load_distribution, sliced_w2
from scripts.diagnose_synthetic import layer_diagnostics, time_diagnostics, digest


def validate_complete(folder,cfg,kind,dataset):
    assert (folder/'COMPLETE').read_text().strip()==str(cfg['steps'])
    assert json.loads((folder/'config.json').read_text())==cfg
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    assert rows[-1]['step']==cfg['steps']
    assert (folder/'checkpoints'/f'step_{cfg["steps"]:07d}.pt').is_file()
    provenance=json.loads((folder/'provenance.json').read_text())
    assert (provenance['dataset'],provenance['model'],provenance['seed'])==(dataset,kind,0)
    return rows,provenance


def diagnose(folder,cfg,entry,dataset,step,output,device,root):
    rows,provenance=validate_complete(folder,cfg,entry['model'],dataset)
    snap=folder/'checkpoints'/f'step_{step:07d}.pt'
    state=torch.load(snap,map_location=device,weights_only=False)
    assert state['step']==step and state['identity']=={'dataset':dataset,'model':entry['model'],'seed':0}
    assert state['config_sha256']==entry['sha256']==digest(folder/'config.json')
    mf=root/cfg['data_root']/'manifest.json'
    assert digest(mf)==state['dataset_manifest_sha256']==provenance['dataset_manifest_sha256']
    # Every training-module source must match the frozen stage code.
    for name,sha in provenance['source_sha256'].items():
        assert digest(root/'synthetic'/name)==sha, f'training source changed: {name}'
    checkpoint_sha=digest(snap)
    if output.exists():
        previous=json.loads(output.read_text())
        assert previous['checkpoint_sha256']==checkpoint_sha
        assert previous['evaluation_script_sha256']==digest(__file__)
        assert previous['helper_sha256']==digest(root/'scripts/diagnose_synthetic.py')
        return
    spec=json.loads(mf.read_text())['datasets'][dataset]
    model=VelocityNet(spec['dim'],entry['model'],**cfg['model_options']).to(device)
    model.load_state_dict(state['model'],strict=True);model.eval()
    fixed=fixed_eval_data(root/cfg['data_root']/dataset,load_distribution(spec),spec['dim'],0,device,
                          cfg['sample_count'],cfg['sliced_directions'])
    samples={n:generated_samples(model,fixed['sample_source'],n) for n in (8,16,32,64,128,256)}
    sampling={str(n):{'nfe':2*n,'sw2':sliced_w2(xx,fixed['sample_target'],fixed['directions']),
                        'endpoint_rms_to_heun256':(xx-samples[256]).square().mean().sqrt().item()}
              for n,xx in samples.items()}
    metric=next(r for r in rows if r['step']==step)
    assert abs(metric['sample']['sliced_w2']-sampling['32']['sw2'])<5e-4
    result={'dataset':dataset,'label':entry['label'],'model':entry['model'],'orders':entry['orders'],
            'step':step,'seed':0,'parameters':entry['parameters'],'forward_macs':entry['forward_macs'],
            'checkpoint_sha256':checkpoint_sha,'config_sha256':entry['sha256'],
            'run_dir':str(folder.resolve()),'training_gpu':provenance['gpu_name'],
            'training_job':provenance['slurm_job_id'],'source_sha256':provenance['source_sha256'],
            'evaluation_script_sha256':digest(__file__),'helper_sha256':digest(root/'scripts/diagnose_synthetic.py'),
            'final_oracle_mse':metric['validation_oracle_mse'],'final_pair_mse':metric['validation_pair_mse'],
            'sampling':sampling,'time_bins':time_diagnostics(model,fixed),
            'layers':layer_diagnostics(model,fixed['probe_x'],fixed['probe_t'])}
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    os.replace(temp,output)
    print(json.dumps({'diagnosed':entry['label'],'dataset':dataset,'step':step,'sw2':sampling['32']['sw2']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output-root',type=Path,default=Path('../runs/synthetic'))
    parser.add_argument('--report-root',type=Path,default=Path('docs/research-2026-10-20'))
    args=parser.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'configs/synthetic/stage_20261020/manifest.json').read_text())
    dataset=manifest['datasets'][args.index]
    # Rotate order across dataset blocks; each dataset's comparisons stay on one GPU.
    ordered=[]
    for wave in ('A','B'):
        entries=[e for e in manifest['entries'] if e['wave']==wave]
        offset=args.index%len(entries)
        ordered+=entries[offset:]+entries[:offset]
    metadata={'dataset':dataset,'job':os.environ.get('SLURM_JOB_ID'),'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'),
              'gpu':torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
              'run_order':[e['label'] for e in ordered],
              'manifest_sha256':digest(root/'configs/synthetic/stage_20261020/manifest.json'),
              'runner_sha256':digest(__file__)}
    args.report_root.mkdir(parents=True,exist_ok=True)
    meta=args.report_root/f'job-{dataset}.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        assert old['manifest_sha256']==metadata['manifest_sha256']
    else:meta.write_text(json.dumps(metadata,indent=2)+'\n')
    for entry in ordered:
        config=root/entry['config'];assert digest(config)==entry['sha256']
        cfg=json.loads(config.read_text())
        folder=args.output_root/cfg['experiment_id']/dataset/entry['model']/'seed000'
        failure=args.report_root/'failures'/dataset/(entry['label']+'.json')
        if failure.exists():
            assert json.loads(failure.read_text())['config_sha256']==entry['sha256']
            continue
        try:
            if folder.exists():validate_complete(folder,cfg,entry['model'],dataset)
            else:
                run(SimpleNamespace(config=str(config),dataset=dataset,model=entry['model'],seed=0,
                                    device=args.device,output_root=str(args.output_root),steps=None,dry_run=False,resume=None))
        except FloatingPointError as error:
            # A scientific nonfinite outcome must not suppress the other controls.
            failure.parent.mkdir(parents=True,exist_ok=True)
            failure.write_text(json.dumps({'dataset':dataset,'label':entry['label'],
                'config_sha256':entry['sha256'],'reason':str(error),'run_dir':str(folder.resolve())},indent=2)+'\n')
            print(json.dumps({'failed':entry['label'],'reason':str(error)}),flush=True)
            continue
        for step in entry['evaluate_steps']:
            diagnose(folder,cfg,entry,dataset,step,
                     args.report_root/'diagnostics'/dataset/f'{entry["label"]}-step{step:07d}.json',args.device,root)
    # Extended baselines share the same first5000 steps; audit technical replication.
    checks={}
    for family in ('silu','swiglu'):
        selected={e['label']:e for e in ordered if e['label'] in (family+'_5000',family+'_budget')}
        states=[]
        for label,e in selected.items():
            cfg=json.loads((root/e['config']).read_text())
            folder=args.output_root/cfg['experiment_id']/dataset/e['model']/'seed000'
            checkpoint=folder/'checkpoints/step_0005000.pt'
            if not checkpoint.exists():break
            states.append(torch.load(checkpoint,weights_only=False,map_location='cpu')['model'])
        if len(states)!=2:
            checks[family]={'available':False,'reason':'5000-step checkpoint missing; see failure records'}
            continue
        diff=max((states[0][k]-states[1][k]).abs().max().item() for k in states[0])
        checks[family]={'model_max_abs_difference_at5000':diff,'within_1e-6':diff<=1e-6}
    (args.report_root/f'prefix-check-{dataset}.json').write_text(json.dumps(checks,indent=2)+'\n')
    (args.report_root/f'FINISHED-{dataset}').write_text('All planned entries processed; inspect completion and failure records separately.\n')

if __name__=='__main__':main()
