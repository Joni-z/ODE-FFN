"""Frozen K2 mean-return diagnosis and independent native-matrix replay."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.mean_return import MeanReturnNet
from synthetic.recurrent_depth import build
from synthetic.train import restore_model

DOC = 'docs/research-mean-return-2026-10-26'
ARMS = {'ode_k2_native_final': 'ode_k2', 'swiglu192_native_final': 'swiglu192',
        'swiglu128_native_deep': 'swiglu128'}
TIMES = (.1, .3, .5, .7, .9)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derivative(model, x, t, capture=None):
    xx = x.detach().clone().requires_grad_(True)
    prediction = model(xx, t, capture=capture)
    jac = torch.stack([torch.autograd.grad(prediction[:, j].sum(), xx,
        retain_graph=j+1<prediction.shape[1])[0] for j in range(prediction.shape[1])], dim=1)
    return prediction.detach().numpy(), jac.detach().numpy()


def native_derivative(model, x, t):
    # Different differentiation path: vectorize single-input Jacobians.
    def one(state, clock):
        return model(state.unsqueeze(0), clock.unsqueeze(0))[0]
    with torch.no_grad():
        value = model(x, t).numpy()
    jac = torch.vmap(torch.func.jacrev(one, argnums=0))(x, t).detach().numpy()
    return value, jac


def field_prediction(model, x, t):
    values = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            values.append(model(torch.from_numpy(x[start:start+1024]),
                torch.from_numpy(t[start:start+1024])).numpy())
    return np.concatenate(values)


def independent_model(parent, variant):
    model = copy.deepcopy(parent)
    hooks = []
    if variant == 'drop_uniform':
        def remove_mean(module, args, value):
            u = torch.ones(value.shape[-1], dtype=value.dtype)/math.sqrt(value.shape[-1])
            return value-(value@u).unsqueeze(-1)*u
        hooks = [b.ffn.register_forward_hook(remove_mean) for b in model.raw.blocks]
    elif variant in ('drop_return', 'drop_secondorder'):
        with torch.no_grad():
            for block in model.raw.blocks:
                layer = block.ffn.ode2
                if variant == 'drop_return':
                    # Actual original forward with CP, rather than output-term subtraction.
                    p = layer.proj.weight
                    u = torch.ones(len(p), dtype=p.dtype)/math.sqrt(len(p))
                    p.copy_((torch.eye(len(p), dtype=p.dtype)-torch.outer(u,u))@p)
                else:
                    layer.orders = 1
    return model, hooks


def run(groups, source, output):
    torch.set_num_threads(1)
    old = load(source/'source-sha256.json')
    for name, value in old.items():
        assert sha(source/name) == value
        assert sha(ROOT/name) == value
    for name, value in load(ROOT.parent/'analysis-source-sha256.json').items():
        assert sha(ROOT/name) == value
    output.mkdir(parents=True, exist_ok=False)
    status = dict(state='RUNNING', pid=os.getpid(), completed=0, total=9)
    write(output/'status.json', status)
    provenance = dict(source_sha256=old, script_sha256=sha(__file__),
        module_sha256=sha(ROOT/'synthetic/mean_return.py'),
        protocol_sha256=sha(ROOT/DOC/'protocol.md'), inputs={})
    metrics, geometry, risk_rows = [], [], []
    max_field = max_jac = 0.0
    field_count = panel_count = 0
    begin = time.monotonic()
    fixed_case = None
    for group in groups:
        assert load(group/'status.json')['state'] == 'COMPLETE'
        assert load(group/'provenance.json')['source_sha256'] == old
        for dataset in load(group/'study.json')['datasets']:
            banks = group/'banks'/dataset
            path = banks/'field.npz'
            provenance['inputs'][str(path)] = sha(path)
            with np.load(path) as data:
                x, t, oracle = [data[k].astype(np.float64) for k in ('x','t','oracle')]
            panels = []
            for clock in TIMES:
                path = banks/f'panel_{clock:.1f}.npz'
                provenance['inputs'][str(path)] = sha(path)
                with np.load(path) as data:
                    panels.append({k: data[k] for k in ('x','t','oracle','oracle_jacobian')})
            for arm, family in ARMS.items():
                folder = output/dataset/arm
                folder.mkdir(parents=True, exist_ok=False)
                original = group/'runs'/dataset/arm
                path = original/'step_0005000.pt'
                provenance['inputs'][str(path)] = sha(path)
                parent = build(x.shape[1], family, 'native')
                restore_model(parent, torch.load(path, map_location='cpu', weights_only=False))
                parent.eval()
                for p in parent.parameters():
                    p.requires_grad_(False)
                wrapper = MeanReturnNet(parent)
                for clock, panel in zip(TIMES, panels):
                    value, jac = derivative(wrapper, torch.from_numpy(panel['x']), torch.from_numpy(panel['t']))
                    path = original/'diagnostics'/f'panel_{clock:.1f}.npz'
                    provenance['inputs'][str(path)] = sha(path)
                    with np.load(path) as data:
                        np.testing.assert_array_equal(value, data['prediction'])
                        np.testing.assert_array_equal(jac, data['jacobian'])
                parent.double()
                variants = ('native','drop_uniform','drop_return','drop_secondorder') if family=='ode_k2' else ('native','drop_uniform')
                native = None
                native_panels = {}
                for variant in variants:
                    wrapper.variant = variant
                    value = field_prediction(wrapper, x, t)
                    assert np.isfinite(value).all()
                    if variant == 'native':
                        native = value
                    if variant == 'drop_uniform':
                        np.testing.assert_allclose(value, native, rtol=1e-10, atol=1e-10)
                    before = np.mean((native-oracle)**2, axis=1)
                    after = np.mean((value-oracle)**2, axis=1)
                    delta = value-native
                    alignment = 2*np.mean((native-oracle)*delta, axis=1)
                    energy = np.mean(delta**2, axis=1)
                    np.testing.assert_allclose(after-before, alignment+energy, rtol=1e-9, atol=1e-12)
                    np.savez_compressed(folder/f'{variant}-field.npz', prediction=value, native_error=before,
                        changed_error=after, error_difference=after-before, alignment=alignment, change_energy=energy)
                    replay, hooks = independent_model(parent, variant)
                    try:
                        independent = field_prediction(replay, x, t)
                        np.testing.assert_allclose(value, independent, rtol=1e-9, atol=1e-10)
                        max_field = max(max_field, float(np.max(np.abs(value-independent))))
                        field_count += 1
                        jac_errors = []
                        for clock, panel in zip(TIMES, panels):
                            px, pt = [torch.from_numpy(panel[k]).double() for k in ('x','t')]
                            trace = [] if variant=='native' and family=='ode_k2' else None
                            pv, pj = derivative(wrapper, px, pt, capture=trace)
                            iv, ij = native_derivative(replay, px, pt)
                            assert np.isfinite(pv).all() and np.isfinite(pj).all()
                            np.testing.assert_allclose(pv, iv, rtol=1e-9, atol=1e-10)
                            np.testing.assert_allclose(pj, ij, rtol=1e-8, atol=1e-9)
                            max_field = max(max_field, float(np.max(np.abs(pv-iv))))
                            max_jac = max(max_jac, float(np.max(np.abs(pj-ij))))
                            if variant=='native':
                                native_panels[clock] = pv, pj
                            if variant=='drop_uniform':
                                np.testing.assert_allclose(pv, native_panels[clock][0], rtol=1e-10, atol=1e-10)
                                np.testing.assert_allclose(pj, native_panels[clock][1], rtol=1e-9, atol=1e-9)
                            base = native_panels[clock][0]
                            panel_before = np.mean((base-panel['oracle'])**2,axis=1)
                            panel_after = np.mean((pv-panel['oracle'])**2,axis=1)
                            panel_alignment = 2*np.mean((base-panel['oracle'])*(pv-base),axis=1)
                            panel_energy = np.mean((pv-base)**2,axis=1)
                            np.testing.assert_allclose(panel_after-panel_before,panel_alignment+panel_energy,rtol=1e-9,atol=1e-12)
                            np.savez_compressed(folder/f'{variant}-panel-{clock:.1f}.npz', prediction=pv, jacobian=pj,
                                native_error=panel_before,changed_error=panel_after,error_difference=panel_after-panel_before,
                                alignment=panel_alignment,change_energy=panel_energy)
                            jac_errors.append(float(np.mean(np.sum((pj-panel['oracle_jacobian'])**2,axis=(1,2)))))
                            panel_count += 1
                            if trace is not None:
                                arrays = {}
                                for record in trace:
                                    index = record['block']
                                    for key in ('z','duration','returning','quadratic'):
                                        arrays[f'b{index}_{key}'] = record[key].detach().numpy()
                                    path_term = record['returning'].detach().numpy()
                                    quad = record['quadratic'].detach().numpy()
                                    local = parent.raw.blocks[index].ffn.ode2
                                    local_cp = copy.deepcopy(local)
                                    with torch.no_grad():
                                        weight = local_cp.proj.weight
                                        weight.sub_(weight.mean(dim=0,keepdim=True))
                                        direct = local(record['z'])-local_cp(record['z'])
                                        direct = direct-direct.mean(-1,keepdim=True)
                                    np.testing.assert_allclose(path_term,direct.numpy(),rtol=1e-9,atol=1e-10)
                                    cq = quad-quad.mean(axis=1, keepdims=True)
                                    p = parent.raw.blocks[index].ffn.ode2.proj.weight.detach().numpy()
                                    u = np.ones(len(p))/math.sqrt(len(p))
                                    b = p@u
                                    b = b-b.mean()
                                    geometry.append(dict(dataset=dataset,time=clock,block=index,
                                        return_energy=float(np.mean(path_term**2)),centered_quadratic_energy=float(np.mean(cq**2)),
                                        return_operator_norm=float(np.linalg.norm(b)*np.linalg.norm(u@p))))
                                np.savez_compressed(folder/f'native-terms-{clock:.1f}.npz', **arrays)
                            if dataset=='rotated_product_bimodal_16d' and family=='ode_k2' and clock==.5:
                                if fixed_case is None:
                                    fixed_case = dict(selection='Product16 ODE t=.5 input0 block1, fixed before outcomes')
                                fixed_case[variant+'_field_error'] = float(np.mean((pv[0]-panel['oracle'][0])**2))
                                if trace is not None:
                                    fixed_case['return_term_rms'] = float(trace[1]['returning'][0].detach().square().mean().sqrt())
                    finally:
                        for hook in hooks:
                            hook.remove()
                    metrics.append(dict(dataset=dataset,arm=arm,variant=variant,primary_mse=float(after.mean()),
                        jacobian_error=float(np.mean(jac_errors))))
                    risk_rows.append(dict(dataset=dataset,arm=arm,variant=variant,
                        error_difference=float((after-before).mean()),alignment=float(alignment.mean()),change_energy=float(energy.mean())))
                status['completed'] += 1
                status['latest'] = [dataset,arm]
                write(output/'status.json', status)
                process = psutil.Process()
                cpu = process.cpu_times()
                with (output/'monitor.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(pid=os.getpid(),completed=status['completed'],elapsed_seconds=time.monotonic()-begin,
                        cpu_seconds=cpu.user+cpu.system,rss_bytes=process.memory_info().rss,
                        available_memory_bytes=psutil.virtual_memory().available))+'\n')
                print(json.dumps(status), flush=True)
    assert status['completed']==9 and field_count==24 and panel_count==120
    for path, value in provenance['inputs'].items():
        assert sha(path)==value
    for name, value in load(ROOT.parent/'analysis-source-sha256.json').items():
        assert sha(ROOT/name)==value
    table(output/'comparison.csv', metrics)
    table(output/'native-paths.csv', geometry)
    table(output/'risk-polarizations.csv', risk_rows)
    comparisons = {}
    for dataset in sorted({r['dataset'] for r in metrics}):
        rows = [r for r in metrics if r['dataset']==dataset]
        candidate = next(r for r in rows if r['variant']=='drop_return')
        own = next(r for r in rows if r['arm'].startswith('ode') and r['variant']=='native')
        generic = [r for r in rows if r['arm'].startswith('swiglu') and r['variant']=='native']
        comparisons[dataset] = dict(primary_ratio=candidate['primary_mse']/min(r['primary_mse'] for r in generic),
            parent_ratio=candidate['primary_mse']/own['primary_mse'],
            jacobian_ratio=candidate['jacobian_error']/min(r['jacobian_error'] for r in generic))
    nonlinear = ('gmm_ring8_2d','rotated_product_bimodal_16d')
    eligible = (all(r['primary_ratio']<=1.05 for r in comparisons.values())
        and any(comparisons[d]['primary_ratio']<=.9 and comparisons[d]['parent_ratio']<=.95 for d in nonlinear)
        and all(comparisons[d]['jacobian_ratio']<=1.1 for d in nonlinear))
    write(output/'decision.json', dict(candidate=bool(eligible),comparisons=comparisons,
        action='registered sampling checks required' if eligible else 'stop mean-return removal; no training or tuning'))
    write(output/'case-study.json', fixed_case)
    write(output/'provenance.json', provenance)
    write(output/'audit.json', dict(status='PASS',models=9,full_field_arrays=field_count,
        full_panel_predictions_and_jacobians=panel_count,max_prediction_absolute_error=max_field,
        max_jacobian_absolute_error=max_jac,independent_path='actual CP matrix edit or K1 order; vmap(jacrev) differentiation',
        native_float32_saved_panel_replay='bitwise',native_local_path_checks=45,
        risk_polarization='per-input full fields and panels',script_sha256=sha(__file__)))
    datasets = ('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
    titles = ('Gaussian2D','Ring8','Product16')
    plot_specs = [('ode_k2_native_final','native','ODE native'),('ode_k2_native_final','drop_return','ODE return removed'),
        ('ode_k2_native_final','drop_secondorder','ODE all second order removed'),
        ('swiglu192_native_final','native','SwiGLU192'),('swiglu128_native_deep','native','SwiGLU128deep')]
    fig, axes = plt.subplots(1,3,figsize=(12,3.5))
    for ax,dataset,title in zip(axes,datasets,titles):
        values = [next(r['primary_mse'] for r in metrics if r['dataset']==dataset and r['arm']==arm and r['variant']==v)
            for arm,v,label in plot_specs]
        ax.scatter(values,range(len(values)),c=['#D55E00','#E69F00','#CC79A7','#0072B2','#009E73'])
        ax.set_yticks(range(len(values)),[s[2] for s in plot_specs] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set(xscale='log',xlabel='Full validation field MSE',title=title)
        for i,value in enumerate(values):
            ax.annotate(f'{value:.4g}',(value,i),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.margins(x=.4)
    fig.tight_layout()
    fig.savefig(output/'field-comparison.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/'field-comparison.pdf',bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(10,3.4))
    for ax,dataset,title in zip(axes,datasets,titles):
        ratios = []
        for index in range(3):
            rows = [r for r in geometry if r['dataset']==dataset and r['block']==index]
            ratios.append(sum(r['return_energy'] for r in rows)/sum(r['centered_quadratic_energy'] for r in rows))
        ax.bar(range(3),ratios,color='#0072B2')
        ax.set(title=title,xlabel='Second solve block',ylabel='Return / centered quadratic energy',xticks=(0,1,2))
    fig.tight_layout()
    fig.savefig(output/'path-energy.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/'path-energy.pdf',bbox_inches='tight')
    plt.close(fig)
    write(output/'status.json',dict(status,state='COMPLETE',field_arrays=field_count,panel_jacobians=panel_count))
    print(json.dumps(dict(state='COMPLETE',candidate=bool(eligible),max_field=max_field,max_jacobian=max_jac)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run([p.resolve() for p in args.inputs],args.source.resolve(),args.output.resolve())
    except BaseException as error:
        status_path = args.output.resolve()/'status.json'
        if status_path.exists():
            status = load(status_path)
            if status.get('pid')==os.getpid() and status.get('state')=='RUNNING':
                write(status_path,dict(status,state='FAILED',error=repr(error)))
        raise
