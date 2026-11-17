"""Reconstruct all-source reference errors and retained GPU prefix parity."""
import argparse
from pathlib import Path
import numpy as np
from population_resolution import read, dump, sha


def check(x, y, atol, rtol):
    error = np.abs(x-y)
    failed = error > atol+rtol*np.abs(y)
    return dict(qualified=bool(not failed.any()), max_error=float(error.max()),
                failed_components=int(failed.sum()))


def main():
    p = argparse.ArgumentParser()
    for name in ('input','main','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists()
    done, main = read(a.input/'COMPLETE.json'), read(a.main/'COMPLETE.json')
    assert done['passed'] and done['tensors_methods_rng_unchanged']
    assert done['source_main_complete_sha256'] == sha(a.main/'COMPLETE.json')
    assert done['source_main_array_sha256'] == sha(a.main/main['file'])
    assert sha(a.input/done['file']) == done['sha256']
    with np.load(a.input/done['file']) as z:
        values = {k:z[k].copy() for k in z.files}
    with np.load(a.main/main['file']) as z:
        original = {k:z[k].copy() for k in z.files}
    assert all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values())
    points = done['points_per_call']
    assert points == main['points'] and done['total_inputs'] == 16*points
    prefix = min(8, points)
    indices = np.concatenate([np.arange(i*points,i*points+prefix) for i in range(16)])
    assert np.array_equal(values['time'][indices], original['probe_t'])
    assert np.array_equal(values['x'].reshape(16,points,-1)[::2], original['augmented_path'][:8,:,:original['native'].shape[1]])
    checks = dict(probe_input=check(values['x'][indices],original['probe_x'],1e-10,1e-9))
    for name,atol,rtol in [('primal',1e-7,1e-7),('tangent',1e-8,1e-5)]:
        checks[name+'_refinement'] = check(values[name+'128'],values[name+'256'],atol,rtol)
        for reference in (128,256):
            key=name+str(reference)
            checks['gpu_'+key] = check(values[key][indices],original['probe_'+key],1e-9,1e-8)
    assert checks == done['checks']
    qualified=all(v['qualified'] for v in checks.values())
    assert qualified == done['numerically_qualified']
    dump(a.output,dict(passed=True,case=done['case'],split=done['split'],bank=done['bank'],
        steps=done['steps'],total_inputs=done['total_inputs'],witness=done['witness'],
        numerically_qualified=qualified,checks_reconstructed=len(checks),
        complete_sha256=sha(a.input/'COMPLETE.json'),source_sha256=sha(__file__)))
    print('INDEPENDENT_ALL_SOURCE_REFERENCE_PASS',done['total_inputs'],qualified,flush=True)


if __name__=='__main__':
    main()
