"""All-source reference refinement at the prescribed native Heun stages."""
import argparse
import functools
import os
from pathlib import Path
import resource
import time
from population_resolution import read, dump, sha, load_model

OWNED_OUTPUT = None


def compare(actual, reference, atol, rtol):
    import numpy as np
    error = np.abs(actual-reference)
    failed = error > atol+rtol*np.abs(reference)
    return dict(qualified=bool(not failed.any()), max_error=float(error.max()),
                failed_components=int(failed.sum()))


def main():
    global OWNED_OUTPUT
    p = argparse.ArgumentParser()
    for name in ('root', 'input', 'output', 'execution-proof'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4096)
    a = p.parse_args()
    assert not a.output.exists() and 0 < a.batch_size <= 4096
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    import numpy as np
    import torch
    from common import base, ev
    from compiled_defect_observer import response_with_step
    from factored_defect_drift import coupled_step
    torch.set_num_threads(1)
    base.verify_sources()
    done = read(a.input/'COMPLETE.json')
    independent = read(a.input/'independent.json')
    assert done['passed'] and independent['passed']
    assert independent['complete_sha256'] == sha(a.input/'COMPLETE.json')
    assert done['portable_sha256'] == sha(a.root/'portable.json')
    bankfile = a.input/done['file']
    assert sha(bankfile) == done['sha256']
    proof = read(a.execution_proof)
    assert proof['passed'] and not proof['witness'] and proof['variant'] == 'power-reuse'
    drift_file = Path(__file__).with_name('factored_defect_drift.py')
    assert proof['step_source_sha256'] == sha(drift_file)
    a.output.mkdir(parents=True)
    OWNED_OUTPUT = a.output
    began = time.monotonic()
    with np.load(bankfile) as data:
        values = {k: data[k].copy() for k in data.files}
    points = done['points']
    assert points == (2 if done['witness'] else 4096)
    case = next(c for c in read(a.root/'portable.json')['cases'] if c['id'] == done['case'])
    model, _ = load_model(a.root, case)
    snapshot = ev.tensor_snapshot(model)
    methods = [layer.solve.__func__ for _,layer in model.ode_layers()]
    rng = torch.get_rng_state().clone()
    dim = values['native'].shape[1]
    path = values['augmented_path'][:8, :, :dim]
    assert path.shape == (8, points, dim)
    stages, times = [], []
    h = 1/done['steps']
    with torch.no_grad():
        for k, array in enumerate(path):
            x = torch.from_numpy(array.copy())
            t = torch.full((points, 1), k/8, dtype=x.dtype)
            predictor = x+h*model(x, t)
            stages += [x.numpy().copy(), predictor.numpy().copy()]
            times += [t.numpy().copy(), torch.full_like(t, k/8+h).numpy()]
    x, t = np.concatenate(stages), np.concatenate(times)
    prefix = min(8, points)
    indices = np.concatenate([np.arange(i*points, i*points+prefix) for i in range(16)])
    reconstruction = compare(x[indices], values['probe_x'], 1e-10, 1e-9)
    assert np.array_equal(t[indices], values['probe_t'])
    assert done['probe_calls'] == [2*(k*(done['steps']//8))+stage for k in range(8) for stage in (0,1)]
    step = functools.partial(coupled_step, reuse_powers=True)
    outputs, timings = {}, []
    for reference in (128, 256):
        primal, tangent = [], []
        for start in range(0, len(x), a.batch_size):
            stop = min(start+a.batch_size, len(x))
            dump(a.output/'progress.json', dict(pid=os.getpid(), case=done['case'], split=done['split'],
                bank=done['bank'], steps=done['steps'], reference=reference,
                completed_inputs=start, total_inputs=len(x), seconds=time.monotonic()-began))
            began_batch = time.perf_counter()
            pp, zz = response_with_step(model, torch.from_numpy(x[start:stop]),
                torch.from_numpy(t[start:stop]), reference,
                case['trained_solver'], case['inference_solver'], step)
            timings.append(dict(reference=reference, points=stop-start,
                                seconds=time.perf_counter()-began_batch))
            primal.append(pp.numpy()); tangent.append(zz.numpy())
        outputs['primal'+str(reference)] = np.concatenate(primal)
        outputs['tangent'+str(reference)] = np.concatenate(tangent)
    checks = dict(probe_input=reconstruction)
    for name, atol, rtol in [('primal',1e-7,1e-7), ('tangent',1e-8,1e-5)]:
        checks[name+'_refinement'] = compare(outputs[name+'128'], outputs[name+'256'], atol, rtol)
        for reference in (128,256):
            key = name+str(reference)
            checks['gpu_'+key] = compare(outputs[key][indices], values['probe_'+key], 1e-9, 1e-8)
    out = a.output/'probes.npz'
    np.savez_compressed(out, x=x, time=t, **outputs)
    ev.same_tensors(model, snapshot)
    assert methods == [layer.solve.__func__ for _,layer in model.ode_layers()]
    assert torch.equal(rng, torch.get_rng_state())
    result = dict(passed=True, witness=done['witness'], case=done['case'], split=done['split'],
        bank=done['bank'], steps=done['steps'], points_per_call=points, calls=16,
        total_inputs=len(x), batch_size=a.batch_size, checks=checks,
        numerically_qualified=all(v['qualified'] for v in checks.values()),
        source_main_complete_sha256=sha(a.input/'COMPLETE.json'),
        source_main_array_sha256=sha(bankfile), source_main_independent_sha256=sha(a.input/'independent.json'),
        portable_sha256=sha(a.root/'portable.json'), file=out.name, sha256=sha(out),
        source_sha256=sha(__file__), drift_source_sha256=sha(drift_file),
        execution_proof_sha256=sha(a.execution_proof), tensors_methods_rng_unchanged=True,
        dtype='float64', device='cpu', timings=timings,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        seconds=time.monotonic()-began)
    dump(a.output/'COMPLETE.json', result)
    print('ALL_SOURCE_REFERENCE_COMPLETE', done['case'], done['split'], done['bank'],
          done['steps'], len(x), result['numerically_qualified'], result['seconds'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        if OWNED_OUTPUT is not None:
            dump(OWNED_OUTPUT/'FAILED.json',dict(passed=False,error_type=type(error).__name__,
                error=str(error),traceback=traceback.format_exc(),source_sha256=sha(__file__)))
        raise
