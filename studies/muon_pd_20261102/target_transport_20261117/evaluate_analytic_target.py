"""One original bank: analytic/measured forcings through the unchanged Heun."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from population_resolution import read, dump, sha, load_model, select_gpu

OWNED_OUTPUT = None


def main():
    global OWNED_OUTPUT
    p = argparse.ArgumentParser()
    for name in ('root', 'reference', 'output', 'cache', 'execution-proof'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--split', choices=('selection', 'confirmation'), required=True)
    p.add_argument('--bank', type=int, choices=range(8), required=True)
    p.add_argument('--steps', type=int, choices=(128, 2048), default=2048)
    p.add_argument('--gpu-uuid')
    p.add_argument('--witness', action='store_true')
    p.add_argument('--readiness', type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    physical, uuid = select_gpu(a.gpu_uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(a.cache/'inductor')
    os.environ['TRITON_CACHE_DIR'] = str(a.cache/'triton')
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    import numpy as np
    import torch
    from common import base, ev
    from gpu_resolution import exclusive_gpu
    from analytic_target import JointSensitivityField
    from check_analytic_target import check
    from compiled_defect_observer import response_with_step
    from factored_defect_drift import make_compiled_step
    from learned_transport import integrate, squared_sw2, target_derivative
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    base.verify_sources()
    proof = read(a.execution_proof)
    assert proof['passed'] and not proof['witness'] and proof['rows'] == 8
    assert proof['median_speedup'] > 1
    variant = proof['variant']
    assert variant in ('linear-factorization', 'power-reuse')
    assert proof['step_source_sha256'] == sha(Path(__file__).with_name('factored_defect_drift.py'))
    verified = read(a.execution_proof.with_name('independent.json'))
    assert verified['passed'] and verified['complete_sha256'] == sha(a.execution_proof)
    if not a.witness:
        ready = read(a.readiness)
        assert ready['passed'] and ready['source_sha256'] == sha(__file__)
        assert ready['portable_sha256'] == sha(a.root/'portable.json')
    guard = lambda: exclusive_gpu(uuid)
    guard()
    a.output.mkdir(parents=True)
    OWNED_OUTPUT = a.output
    began = time.monotonic()
    portable = read(a.root/'portable.json')
    case = next(c for c in portable['cases'] if c['id'] == a.case)
    native, crossed = (m.cuda() for m in load_model(a.root, case))
    snapshots = [ev.tensor_snapshot(m) for m in (native, crossed)]
    methods = [layer.solve.__func__ for _, layer in native.ode_layers()]
    rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    binding = case['splits'][a.split]['assets']
    asset_file = a.root/binding['file']
    assert sha(asset_file) == binding['sha256']
    count, steps, hidden_steps = (2, 8, 16) if a.witness else (4096, a.steps, 256)
    with np.load(asset_file) as data:
        source = torch.from_numpy(data['source_'+str(a.bank)][:count].copy()).double().cuda()
        target = data['target_'+str(a.bank)][:count].astype(np.float64)
        directions = data['directions'].astype(np.float64)
    n, c = case['trained_solver'], case['inference_solver']
    step = make_compiled_step(variant == 'power-reuse')

    def response(x, t, reference=hidden_steps):
        return response_with_step(native, x, t, reference, n, c, step)

    # Compilation may consume RNG; isolate warmup, then require exact replay.
    warm_start = time.monotonic()
    with torch.random.fork_rng(devices=[0]):
        response(source, torch.zeros((count, 1), dtype=source.dtype, device='cuda'))
        torch.cuda.synchronize()
    warm_seconds = time.monotonic()-warm_start
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    prefix_derivative = None
    if a.witness:
        with torch.random.fork_rng(devices=[0]):
            prefix_derivative = check(native, crossed, lambda x,t: response(x,t,16)[1],
                                      source, 8, analytic_jvp=False)
    probe_calls = {2*(k*(steps//8))+stage for k in range(8) for stage in (0, 1)}
    probes = []

    def observe(call, x, t, analytic):
        if call in probe_calls:
            probes.append((call, x[:8].cpu().numpy(), t[:8].cpu().numpy(), analytic[:8].cpu().numpy()))
        if call % 8 == 0:
            dump(a.output/'progress.json', dict(pid=os.getpid(), case=a.case, split=a.split,
                bank=a.bank, call=call, total_calls=2*steps, steps=steps,
                seconds=time.monotonic()-began, stage='analytic-target-trajectory'))
        if call % 64 == 0:
            guard()
            status = subprocess.check_output(['nvidia-smi', '-i', uuid,
                '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw', '--format=csv,noheader'], text=True)
            import json
            with (a.output/'gpu-history.jsonl').open('a') as f:
                f.write(json.dumps(dict(unix_time=time.time(), call=call, resource=status.strip()))+'\n')

    joint = JointSensitivityField(native, crossed, lambda x,t: response(x,t)[1], steps, observe)
    initial = torch.cat((source, torch.zeros_like(source), torch.zeros_like(source)), dim=1)
    run_start = time.monotonic()
    with torch.no_grad():
        endpoint, path = integrate(base.train.generated_samples, joint, initial, steps)
    torch.cuda.synchronize()
    trajectory_seconds = time.monotonic()-run_start
    assert joint.calls == 2*steps and len(probes) == 16
    dim = source.shape[1]
    x, ua, um = (v.cpu().numpy() for v in endpoint.split(dim, dim=1))
    if a.witness:
        expected = base.train.generated_samples(native, source, steps).cpu().numpy()
        alternate = base.train.generated_samples(crossed, source, steps).cpu().numpy()
        reference_sha = None
    else:
        reference_rows = read(a.reference/a.case/'rows.json')
        row = next(r for r in reference_rows if r['split'] == a.split and r['steps'] == steps)
        reference_file = a.reference/a.case/row['file']
        assert sha(reference_file) == row['sha256']
        with np.load(reference_file) as data:
            expected, alternate = (data[k][a.bank].copy() for k in ('native', 'crossed'))
        reference_sha = sha(reference_file)
    replay_error = np.abs(x-expected)
    assert np.all(replay_error <= 1e-10+1e-9*np.abs(expected))
    px = torch.from_numpy(np.concatenate([v[1] for v in probes])).cuda()
    pt = torch.from_numpy(np.concatenate([v[2] for v in probes])).cuda()
    recorded = np.concatenate([v[3] for v in probes])
    with torch.random.fork_rng(devices=[0]):
        p128, z128 = (v.cpu().numpy() for v in response(px, pt, 128))
        p256, z256 = (v.cpu().numpy() for v in response(px, pt, 256))
        replay_b = response(px, pt, hidden_steps)[1].cpu().numpy()
    gates = {}
    for name, actual, reference, atol, rtol in [
        ('observer_primal_refinement', p128, p256, 1e-7, 1e-7),
        ('observer_tangent_refinement', z128, z256, 1e-8, 1e-5),
        ('observer_probe_replay', recorded, replay_b, 1e-10, 1e-8)]:
        error = np.abs(actual-reference)
        gates[name] = dict(qualified=bool(np.all(error <= atol+rtol*np.abs(reference))),
                           max_error=float(error.max()))
    loss0, loss1 = (squared_sw2(v, target, directions) for v in (expected, alternate))
    delta = loss1-loss0
    predictions = {}
    for name, tangent in [('analytic', ua), ('measured', um)]:
        derivative = target_derivative(x, tangent, target, directions)
        predicted = derivative['loss_derivative']
        predictions[name] = dict(derivative=derivative, actual_loss_change=delta,
            score=None if delta == 0 else 1-(predicted-delta)**2/delta**2,
            sign_agrees=bool(np.sign(predicted) == np.sign(delta)),
            endpoint_metric_change=squared_sw2(x+tangent, target, directions)-loss0)
    file = a.output/'bank.npz'
    np.savez_compressed(file, native=x, analytic_tangent=ua, measured_tangent=um,
        native_reference=expected, crossed_reference=alternate, target=target,
        directions=directions, augmented_path=path.cpu().numpy(),
        probe_x=px.cpu().numpy(), probe_t=pt.cpu().numpy(), probe_recorded=recorded,
        probe_replayed=replay_b,
        probe_primal128=p128, probe_primal256=p256, probe_tangent128=z128, probe_tangent256=z256)
    for model, snapshot in zip((native, crossed), snapshots, strict=True):
        ev.same_tensors(model, snapshot)
    assert methods == [layer.solve.__func__ for _, layer in native.ode_layers()]
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    counters = {k: {str(x): int(y) for x,y in v.items()} for k,v in torch._dynamo.utils.counters.items()}
    assert not counters.get('graph_break') and not counters.get('unimplemented')
    qualified = joint.primal_qualified and all(v['qualified'] for v in gates.values())
    qualified &= all(v['derivative']['qualified'] for v in predictions.values())
    result = dict(passed=True, witness=a.witness, case=a.case, split=a.split, bank=a.bank,
        points=count, steps=steps, hidden_reference_steps=hidden_steps,
        physical_gpu=physical, gpu_uuid=uuid, variant=variant,
        numerically_qualified=bool(qualified), primal_jvp_max_error=joint.primal_max_error,
        native_replay_max_error=float(replay_error.max()), numerical=gates,
        predictions=predictions, prefix_derivative=prefix_derivative,
        native_loss=loss0, crossed_loss=loss1, file=file.name, sha256=sha(file),
        probe_calls=[v[0] for v in probes],
        reference_sha256=reference_sha, asset_sha256=sha(asset_file),
        portable_sha256=sha(a.root/'portable.json'), source_sha256=sha(__file__),
        kernel_sha256=sha(Path(__file__).with_name('analytic_target.py')),
        execution_proof_sha256=sha(a.execution_proof), compiler_counters=counters,
        tensors_methods_rng_unchanged=True, warm_seconds=warm_seconds,
        trajectory_seconds=trajectory_seconds, seconds=time.monotonic()-began)
    dump(a.output/'COMPLETE.json', result)
    print('ANALYTIC_TARGET_BANK_COMPLETE', a.case, a.split, a.bank, steps, qualified,
          trajectory_seconds, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        if OWNED_OUTPUT is not None:
            dump(OWNED_OUTPUT/'FAILED.json', dict(passed=False, error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), source_sha256=sha(__file__)))
        raise
