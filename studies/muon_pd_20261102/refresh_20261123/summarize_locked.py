"""Audit the locked cohort and give paired differences for the two benchmarks."""
from pathlib import Path
import sys
import numpy as np
from final_banks import HERE, DATASETS, base, read


def collect(work):
    spec=read(HERE.parents[2]/'docs/report4/midpoint-refresh/bounded-replication-lock.json')
    complete=read(work/'results.json')
    models=read(work/'endpoints.json')
    rows=complete['rows']
    assert complete['stream']==spec['stream'] and len(rows)==len(models)==12
    assert [{k:v for k,v in m.items() if k!='folder'} for m in models]==spec['weights']
    output=[]
    for ds in DATASETS:
        subset=[r for r in rows if r['model']['dataset']==ds]
        assert len(subset)==6
        candidate,=[r for r in subset if r['model']['label']=='candidate']
        for r in subset:
            assert (work/ds/r['model']['label']/'score.json').is_file()
            assert r['bank_hashes']==candidate['bank_hashes']==read(work/f'{ds}-banks.json')['array_hashes']
            assert len(r['bank_sw2'])==8
            assert r['model'] in models
        controls=[r for r in subset if r is not candidate]
        archived=[r for r in controls if not r['model']['label'].endswith('-symmetry')]
        best=min(archived,key=lambda r:r['pooled_sw2'])
        symmetry,=[r for r in controls if r not in archived]
        def paired(r):
            delta=np.array(candidate['bank_sw2'])-np.array(r['bank_sw2'])
            return dict(label=r['model']['label'],family=r['model']['family'],checkpoint_sha256=r['model']['checkpoint_sha256'],
                        step=r['model']['step'],sw2=r['pooled_sw2'],pooled_delta=candidate['pooled_sw2']-r['pooled_sw2'],
                        paired_mean_delta=delta.mean(),banks_favoring_candidate=int((delta<0).sum()))
        output.append(dict(dataset=ds,training_seed=0,stream=spec['stream'],banks=8,samples_per_bank=4096,
                           directions=256,outer_heun=128,ode_step=candidate['model']['step'],ode_sw2=candidate['pooled_sw2'],
                           ode_checkpoint_sha256=candidate['model']['checkpoint_sha256'],
                           best_archived_control=paired(best),symmetry_control=paired(symmetry),
                           all_controls=[paired(r) for r in controls]))
    return output


if __name__=='__main__':
    base.dump(Path(sys.argv[2]),collect(Path(sys.argv[1])))
