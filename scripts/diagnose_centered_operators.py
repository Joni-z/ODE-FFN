"""Read-only raw P versus gamma*I+cP spectra in the location factorial."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(matrix):
    singular=np.linalg.svd(matrix,compute_uv=False)
    eigen=np.linalg.eigvals(matrix)
    total=float((singular**2).sum())
    return dict(sigma_max=float(singular[0]),sigma_min=float(singular[-1]),
        singular_values=singular.tolist(),singular_cv=float(singular.std()/max(singular.mean(),1e-15)),
        frobenius=float(np.sqrt(total)),stable_rank=total/max(float(singular[0]**2),1e-30),
        eigen_real_min=float(eigen.real.min()),eigen_real_max=float(eigen.real.max()),
        spectral_radius=float(np.abs(eigen).max()),
        nonnormality=float(np.linalg.norm(matrix.T@matrix-matrix@matrix.T)/(total+1e-15)))


def diagnose(study,output):
    torch.set_num_threads(1)
    status=json.loads((study/'status.json').read_text())
    assert status['state']=='COMPLETE'
    audit=json.loads((study/'audit/audit.json').read_text())
    assert audit['state']=='PASS' and audit['source_verified']
    provenance=json.loads((study/'provenance.json').read_text())
    rows=[];hashes={};initials={};count=0
    for dataset in provenance['plan']['datasets']:
        initials[dataset]={}
        for arm in provenance['plan']['arms']:
            if arm['model'] not in ('ode_original','ode_vector'):continue
            label=arm['label'];config=study/'configs'/f'{label}.json'
            cfg=json.loads(config.read_text());options=cfg['model_options']
            assert options['orders']==1
            modes=options['ode_solve_pattern'].split('_')
            center=options['shift']+options['scale']/2
            root=study/'runs'/cfg['experiment_id']/dataset/arm['model']/'seed000/checkpoints'
            checkpoints=sorted(root.glob('step_*.pt'))
            assert len(checkpoints)==6
            for path in checkpoints:
                state=torch.load(path,map_location='cpu',weights_only=False)
                assert state['config_sha256']==sha(config)
                assert state['dataset_manifest_sha256']==provenance['dataset_manifest_sha256']
                assert state['identity']==dict(dataset=dataset,model=arm['model'],seed=0)
                hashes[str(path.relative_to(study))]=sha(path);count+=1
                for name,value in state['model'].items():
                    if not name.endswith('.proj.weight'):continue
                    matrix=value.detach().numpy().astype(np.float64)
                    index=0 if '.ode1.' in name else 1
                    gamma=int(modes[index]=='map')
                    anchored=gamma*np.eye(len(matrix))+center*matrix
                    key=name.removesuffix('.proj.weight')
                    if state['step']==0:initials[dataset].setdefault(label,{})[key]=anchored
                    rows.append(dict(dataset=dataset,label=label,step=state['step'],layer=key,
                        gamma=gamma,center=center,raw=statistics(matrix),centered=statistics(anchored)))
    maximum=0.
    for dataset,models in initials.items():
        reference=models['scalar_mm']
        assert len(models)==8
        for model in models.values():
            for key,value in model.items():
                error=float(np.abs(value-reference[key]).max());maximum=max(maximum,error)
                assert error<=2e-7,(dataset,key,error)
    result=dict(state='COMPLETE',rows=rows,checkpoints_read=count,
        initial_centered_matrix_max_difference=maximum,initial_comparison_atol=2e-7,
        source_sha256=sha(Path(__file__)),native_audit_sha256=sha(study/'audit/audit.json'),
        checkpoint_sha256=hashes,
        scope='P and the fixed-reference-duration matrix gamma*I+cP. Excludes actual state-dependent durations, controller derivatives and the complete velocity Jacobian. Read-only post-hoc analysis, no training.')
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=diagnose(args.study,args.output)
    print(json.dumps({k:result[k] for k in ('state','checkpoints_read','initial_centered_matrix_max_difference')}))
