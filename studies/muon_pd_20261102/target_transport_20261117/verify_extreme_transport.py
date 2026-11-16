"""Independent NumPy/cache reconstruction of rare-source diagnostic records."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(actual,reference,stored,absolute,relative):
    error=np.abs(actual-reference)
    finite=np.isfinite(actual)&np.isfinite(reference)
    mask=finite&(error<=absolute+relative*np.abs(reference))
    assert bool(mask.all())==stored['qualified']
    assert int(np.sum(~mask))==stored['failed_components']
    assert int(np.sum(~finite))==stored['nonfinite_components']
    if np.isfinite(error).all():assert abs(float(error.max())-stored['max_error'])<1e-12
    else:assert stored['max_error'] is None
    return bool(mask.all())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--input-cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    summary,complete=(read(args.input/p) for p in ('summary.json','COMPLETE.json'))
    assert complete['passed'] and complete['summary_sha256']==sha(args.input/'summary.json')
    binding=summary['binding'];assert binding['input_cases_sha256']==sha(args.input_cases)
    cohort=read(args.input_cases)
    case=next(v for v in cohort['cases'] if v['case']==summary['case'] and v['split']==summary['split'])
    assert sha(case['file'])==case['sha256']==binding['selected_arrays_sha256']
    with np.load(case['file']) as z:cache={k:z[k] for k in z.files}
    old=Path(case['file']).parent.parent/'formal'/case['split']/case['case']
    assert sha(old/'summary.json')==case['original_summary_sha256']
    assert sha(old/'binding.json')==case['original_binding_sha256']
    previous,old_binding=read(old/'summary.json'),read(old/'binding.json')
    assert old_binding['assets']['sha256']==case['original_assets_sha256']
    asset=Path(old_binding['assets']['file']);assert sha(asset)==case['original_assets_sha256']
    with np.load(asset) as source:
        for role_index,role in enumerate(case['roles']):
            bank,index=role['bank'],role['index']
            assert np.array_equal(cache['source32'][role_index],source[f'source_{bank}'][index])
            row=previous['banks'][bank];file=old/row['file']
            assert sha(file)==row['sha256']==role['original_bank_file_sha256']
            with np.load(file) as z:
                assert np.array_equal(cache['cached_native'][role_index],z['native'][index])
                assert np.array_equal(cache['cached_crossed'][role_index],z['endpoints'][-1,index])
                assert np.array_equal(cache['cached_tangent'][role_index],z['tangent'][index])
    identities=[(v['bank'],v['index']) for v in case['roles']]
    unique=list(dict.fromkeys(identities));indices=[identities.index(v) for v in unique]
    assert summary['unique_sources']==len(unique)
    assert all(v['unique_row']==unique.index(identities[i]) for i,v in enumerate(summary['roles']))
    assert sha(args.input/'arrays.npz')==binding['arrays_sha256']
    with np.load(args.input/'arrays.npz') as z:arrays={k:z[k] for k in z.files}
    assert np.array_equal(arrays['source32'],cache['source32'][indices])
    expected=[128] if summary['witness'] else [127,128,129,256,512]
    assert arrays['steps'].tolist()==expected
    assert arrays['epsilons'].tolist()==[1e-4,1e-5,1e-6,1e-7]
    valid=True
    for i,step in enumerate(expected):
        row=summary['steps'][i];assert row['steps']==step
        exact=np.array_equal(arrays['native'][i],arrays['augmented_native'][i])
        assert exact==row['augmented_primal_exact']
        finite=all(np.isfinite(arrays[k][i]).all() for k in ('native','crossed','tangent'))
        assert bool(finite)==row['all_outputs_finite']
        valid &= exact and finite and row['jvp_primal_qualified']
    index=expected.index(128)
    for key,absolute,relative in [('native',2e-8,2e-8),('crossed',2e-8,2e-8),('tangent',2e-6,1e-5)]:
        valid &= compare(arrays[key][index],cache['cached_'+key][indices],
                         summary['cache_checks'][key],absolute,relative)
    for i,epsilon in enumerate(arrays['epsilons']):
        row=summary['finite_differences'][i];assert row['epsilon']==epsilon
        reference=(arrays['fd_plus'][i]-arrays['fd_minus'][i])/(2*epsilon)
        for j,record in enumerate(row['unique_source_comparisons']):
            compare(arrays['tangent'][index,j],reference[j],record,2e-5,1e-3)
    assert bool(valid)==summary['replay_qualified']==complete['replay_qualified']
    assert summary['parameters_buffers_rng_unchanged']
    out=dict(passed=True,replay_qualified=bool(valid),witness=summary['witness'],
        case=summary['case'],split=summary['split'],roles=len(case['roles']),unique_sources=len(unique),
        original_sources_and_cached_arrays_exact=True,all_epsilon_records_reconstructed=True,
        source_sha256=sha(__file__),summary_sha256=sha(args.input/'summary.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print('EXTREME_TRANSPORT_INDEPENDENT_PASS',summary['case'],summary['split'],bool(valid),flush=True)


if __name__=='__main__':main()
