"""Independent finite-clock collapse via folded matrices, not clock patches.

Preparation alone does not evaluate checkpoints. Execute only after the main
diagnostic has completed following its protocol and independent code review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
TIMES = (.1, .3, .5, .7, .9)
LAYERS = tuple(f'blocks.{i}.ffn.ode{j}' for i in range(3) for j in (1, 2))
MODEL_SHA = '9fe326f400862597ac28a392b797f51b406955fdfaf5d373eed3907e71caf337'
PROTOCOL_SHA = 'df6eb527a0f6bd82ab5660251b878f834e9027c38e099e654686f9c1dc221411'
F_TOL, J_TOL, METRIC_ATOL, METRIC_RTOL = 1e-10, 1e-9, 1e-10, 1e-11


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_dict(left, right):
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


class FoldedFFN(nn.Module):
    """Two constant linear maps with the original middle SiLU; no RNG draws."""
    def __init__(self, A, B):
        super().__init__()
        self.register_buffer('A', A.detach().clone())
        self.register_buffer('B', B.detach().clone())

    def forward(self, h):
        return F.linear(F.silu(F.linear(h, self.A)), self.B)


def calculate(model, x, t):
    def one(point, time):
        return model(point.unsqueeze(0), time.unsqueeze(0)).squeeze(0)
    with torch.no_grad():
        prediction = model(x, t)
    derivative = torch.vmap(torch.func.jacrev(one, argnums=0))(x, t)
    return prediction.detach().numpy(), derivative.detach().numpy()


def matrix_contract():
    P = torch.tensor([[.3, -.4], [.7, -.2]], dtype=torch.float64)
    Q = torch.tensor([[-.1, .6], [.2, .5]], dtype=torch.float64)
    s, t = torch.tensor([.25, .9], dtype=P.dtype), torch.tensor([.7, .35], dtype=P.dtype)
    x = torch.tensor([[.8, -.3], [-.4, .6]], dtype=P.dtype)
    rng = torch.get_rng_state().clone()
    model = FoldedFFN(torch.eye(2, dtype=P.dtype)+s[:, None]*P,
                      torch.eye(2, dtype=P.dtype)+t[:, None]*Q)
    torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
    def unfused(z):
        first = F.silu(z+F.linear(z, P)*s)
        return first+F.linear(first, Q)*t
    torch.testing.assert_close(model(x), unfused(x), atol=5e-16, rtol=5e-15)
    torch.testing.assert_close(torch.autograd.functional.jacobian(model, x),
                               torch.autograd.functional.jacobian(unfused, x), atol=5e-16, rtol=5e-15)
    assert not list(model.parameters())


def metrics(prediction, jacobian, collapsed, collapsed_jacobian, oracle, oracle_jacobian):
    residual, delta = prediction-oracle, collapsed-prediction
    jr, jd = jacobian-oracle_jacobian, collapsed_jacobian-jacobian
    original, altered = float(np.mean(residual**2)), float(np.mean((collapsed-oracle)**2))
    align, energy = float(2*np.mean(residual*delta)), float(np.mean(delta**2))
    joriginal = float(np.mean(np.sum(jr*jr, axis=(1, 2))))
    jaltered = float(np.mean(np.sum((collapsed_jacobian-oracle_jacobian)**2, axis=(1, 2))))
    jalign = float(2*np.mean(np.sum(jr*jd, axis=(1, 2))))
    jenergy = float(np.mean(np.sum(jd*jd, axis=(1, 2))))
    return dict(native_field_mse=original, collapsed_field_mse=altered,
                field_risk_change=altered-original, field_alignment_term=align,
                field_change_energy=energy, field_polarization_residual=altered-original-align-energy,
                native_jacobian_error=joriginal, collapsed_jacobian_error=jaltered,
                jacobian_risk_change=jaltered-joriginal, jacobian_alignment_term=jalign,
                jacobian_change_energy=jenergy, jacobian_polarization_residual=jaltered-joriginal-jalign-jenergy)


def main(runtime, analysis):
    runtime, analysis = runtime.resolve(), analysis.resolve()
    assert runtime.name == 'affine_forcing_20261025_m1'
    source = runtime/'source'
    output = Path(__file__).resolve().parent
    assert source not in output.parents and runtime/'output' not in output.parents
    hashes = {}
    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())
    manifest = read(runtime/'source-manifest.json')
    assert len(manifest['all_files']) == 595
    names = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert names == set(manifest['all_files'])
    for name, digest in manifest['all_files'].items(): assert sha(source/name) == digest
    assert sha(source/'synthetic/models.py') == MODEL_SHA
    protocol = analysis/'source/protocol-clock-collapse.md'
    hashes[str(protocol)] = sha(protocol)
    assert hashes[str(protocol)] == PROTOCOL_SHA
    assert read(analysis/'summary.json')['state'] == 'PASS', 'Wait for completed primary diagnosis'
    provenance = read(analysis/'provenance.json')
    assert provenance['protocol_sha256'] == PROTOCOL_SHA
    assert provenance['source_manifest_sha256'] == sha(runtime/'source-manifest.json')
    for name, digest in provenance['output_sha256'].items(): assert sha(name) == digest
    csv_path = analysis/'clock-collapse.csv'; hashes[str(csv_path)] = sha(csv_path)
    csv_rows = list(csv.DictReader(csv_path.open()))
    assert len(csv_rows) == 18
    sys.path.insert(0, str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import restore_model
    for module in ('synthetic.models', 'synthetic.train'):
        assert Path(sys.modules[module].__file__).resolve().is_relative_to(source)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    matrix_contract()
    rows, mean_records = [], []
    maximums = dict(historical_prediction=0., historical_jacobian=0., native_prediction=0., native_jacobian=0., folded_prediction=0., folded_jacobian=0., csv_metric=0.)
    for dataset in DATASETS:
        dim = 16 if dataset == DATASETS[-1] else 2
        suffix = 'product16' if dim == 16 else 'v1'
        study = runtime/'output'/f'affine_forcing_{suffix}_20261025_m1'
        assert read(study/'status.json')['state'] == 'COMPLETE'
        config_path = study/'configs/vector_parent.json'; cfg = read(config_path)
        run = study/'runs'/cfg['experiment_id']/dataset/'ode_vector/seed000'
        checkpoint_path = run/'checkpoints/step_0005000.pt'
        hashes[str(checkpoint_path)] = sha(checkpoint_path)
        cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert cp['identity'] == dict(dataset=dataset, model='ode_vector', seed=0) and cp['step'] == 5000
        assert cp['config_sha256'] == sha(config_path)
        assert cfg['model_options']['gate_hidden'] == 32 and cfg['model_options']['orders'] == 1
        assert cfg['model_options']['ode_forcing_mode'] == 'none'
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('product16_v1' if dim == 16 else 'v1')
        dm = read(data/'manifest.json'); assert cp['dataset_manifest_sha256'] == sha(data/'manifest.json')
        probe_path = data/dataset/'probe.npz'
        hashes[str(probe_path)] = sha(probe_path)
        assert hashes[str(probe_path)] == dm['files'][dataset+'/probe.npz']['sha256']
        with np.load(probe_path, allow_pickle=False) as bank:
            px, pt = bank['xt'][:256].copy(), bank['t'][:256].copy()
        assert px.dtype == pt.dtype == np.float32
        native32 = VelocityNet(dim, 'ode_vector', **cfg['model_options']).eval()
        restore_model(native32, cp)
        before = {k: v.clone() for k, v in native32.state_dict().items()}
        before_buffers = {k: v.clone() for k, v in native32.named_buffers()}
        captured, handles = {}, []
        layers = list(native32.ode_layers()); assert tuple(n for n, _ in layers) == LAYERS
        for name, layer in layers:
            assert layer.forcing_mode == 'none' and layer.orders == 1 and layer.gate_source == 'input'
            assert layer.gate_transform == 'sigmoid' and layer.operator_mode == 'multiplicative'
            def hook(module, args, raw, name=name, tau=layer.tau, scale=layer.scale, shift=layer.shift):
                assert name not in captured, 'Each native controller must execute exactly once'
                captured[name] = (torch.sigmoid(raw.detach()/tau)*scale+shift).clone()
            handles.append(layer.t_func.register_forward_hook(hook))
        rng = torch.get_rng_state().clone()
        try:
            with torch.no_grad(): native32(torch.from_numpy(px), torch.from_numpy(pt))
        finally:
            for handle in handles: handle.remove()
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        exact_dict(before, native32.state_dict()); exact_dict(before_buffers, dict(native32.named_buffers()))
        durations = torch.stack([captured[name] for name in LAYERS])
        assert durations.shape == (6, 256, 64) and durations.dtype == torch.float32
        means = durations.mean(dim=1)
        mean_path = analysis/'clock-means'/f'{dataset}.npz'; hashes[str(mean_path)] = sha(mean_path)
        with np.load(mean_path, allow_pickle=False) as bank:
            assert tuple(bank['layer_names'].tolist()) == LAYERS
            for key, expected in (('means', means.numpy()), ('durations', durations.numpy()), ('probe_x', px), ('probe_t', pt)):
                np.testing.assert_array_equal(bank[key], expected)
        log = run/'metrics.jsonl'; hashes[str(log)] = sha(log)
        final = json.loads(log.read_text().splitlines()[-1]); assert final['step'] == 5000
        logged = {row['layer']: row['activation'] for row in final['ode_layers']}
        for name in LAYERS: assert float(captured[name].mean()) == logged[name]['s_mean']
        native = VelocityNet(dim, 'ode_vector', **cfg['model_options']).double().eval()
        folded = VelocityNet(dim, 'ode_vector', **cfg['model_options']).double().eval()
        restore_model(native, cp); restore_model(folded, cp)
        for model in (native, folded):
            for parameter in model.parameters(): parameter.requires_grad_(False)
            exact_dict({k: v.double() for k, v in cp['runtime_buffers'].items()}, dict(model.named_buffers()))
        original_non_ffn = {k: v.clone() for k, v in folded.state_dict().items() if '.ffn.' not in k}
        original_buffers = {k: v.clone() for k, v in folded.named_buffers()}
        rng = torch.get_rng_state().clone()
        for i, block in enumerate(folded.blocks):
            assert isinstance(block.ffn.act, nn.SiLU) and block.ffn.clock_mode == 'independent'
            P, Q = block.ffn.ode1.proj.weight, block.ffn.ode2.proj.weight
            A = torch.eye(64, dtype=torch.float64)+means[2*i].double()[:, None]*P
            B = torch.eye(64, dtype=torch.float64)+means[2*i+1].double()[:, None]*Q
            block.ffn = FoldedFFN(A, B)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        exact_dict(original_non_ffn, {k: v for k, v in folded.state_dict().items() if '.ffn.' not in k})
        for name, value in original_buffers.items(): torch.testing.assert_close(dict(folded.named_buffers())[name], value, atol=0, rtol=0)
        assert sum(len(list(block.ffn.parameters())) for block in folded.blocks) == 0
        frozen_states = [{k: v.clone() for k, v in model.state_dict().items()} for model in (native, folded)]
        frozen_buffers = [{k: v.clone() for k, v in model.named_buffers()} for model in (native, folded)]
        mean_records.append(dict(dataset=dataset, native_float32_capture_exact=True, probe_rows=256,
                                 means_sha256=hashlib.sha256(means.numpy().tobytes()).hexdigest(),
                                 all_mean_vectors=means.tolist(),fold_constructor_RNG_unchanged=True))
        summary = read(study/'diagnostics'/dataset/'vector_parent/summary.json')
        assert summary['checkpoint_sha256'] == hashes[str(checkpoint_path)]
        assert summary['config_sha256'] == sha(config_path) and summary['manifest_sha256'] == sha(data/'manifest.json')
        validation_path = data/dataset/'validation.npz'; hashes[str(validation_path)] = sha(validation_path)
        assert summary['targets_sha256'] == hashes[str(validation_path)]
        per_task = []
        for time in TIMES:
            original_path = study/'diagnostics'/dataset/'vector_parent'/f'common-{time:.1f}.npz'
            hashes[str(original_path)] = sha(original_path)
            assert hashes[str(original_path)] == summary['arrays_sha256'][original_path.name]
            path = analysis/'arrays'/dataset/f'common-{time:.1f}.npz'; hashes[str(path)] = sha(path)
            with np.load(original_path, allow_pickle=False) as bank:
                old = {key: bank[key] for key in ('x','prediction','jacobian','oracle','oracle_jacobian')}
            with np.load(path, allow_pickle=False) as bank:
                arrays = {key: bank[key] for key in ('x','prediction','jacobian','oracle','oracle_jacobian','collapsed_prediction','collapsed_jacobian')}
            for key in ('x', 'oracle', 'oracle_jacobian'): np.testing.assert_array_equal(old[key], arrays[key])
            for key, tolerance in (('prediction', F_TOL), ('jacobian', J_TOL)):
                difference = float(np.max(np.abs(old[key]-arrays[key])))
                maximums['historical_'+key] = max(maximums['historical_'+key], difference)
                assert difference <= tolerance
            x = torch.from_numpy(arrays['x']); t = torch.full((len(x),1), time, dtype=torch.float64)
            pred, jac = calculate(native,x,t); fpred,fjac = calculate(folded,x,t)
            for name,value,reference,tolerance in (
                ('native_prediction',pred,arrays['prediction'],F_TOL),('native_jacobian',jac,arrays['jacobian'],J_TOL),
                ('folded_prediction',fpred,arrays['collapsed_prediction'],F_TOL),('folded_jacobian',fjac,arrays['collapsed_jacobian'],J_TOL)):
                error = float(np.max(np.abs(value-reference))); maximums[name] = max(maximums[name],error)
                assert error <= tolerance,(dataset,time,name,error)
            record = dict(dataset=dataset,time=time,anchors=len(x),dim=dim,
                **metrics(arrays['prediction'],arrays['jacobian'],arrays['collapsed_prediction'],arrays['collapsed_jacobian'],arrays['oracle'],arrays['oracle_jacobian']))
            rows.append(record); per_task.append(record)
        rows.append(dict(dataset=dataset,time='equal_time_mean',anchors=2560,dim=dim,
                         **{key:float(np.mean([row[key] for row in per_task])) for key in per_task[0] if key not in ('dataset','time','anchors','dim')}))
        for model, state, buffers in zip((native, folded), frozen_states, frozen_buffers):
            exact_dict(state, model.state_dict()); exact_dict(buffers, dict(model.named_buffers()))
    assert len(rows) == 18
    for row in rows:
        matching = next(r for r in csv_rows if r['dataset'] == row['dataset'] and r['time'] == str(row['time']))
        assert int(matching['anchors']) == row['anchors'] and int(matching['dim']) == row['dim']
        for key,value in row.items():
            if key in ('dataset','time','anchors','dim'): continue
            difference = abs(value-float(matching[key])); maximums['csv_metric'] = max(maximums['csv_metric'],difference)
            np.testing.assert_allclose(value,float(matching[key]),atol=METRIC_ATOL,rtol=METRIC_RTOL)
        assert abs(row['field_polarization_residual']) <= METRIC_ATOL*(1+abs(row['field_risk_change']))
        assert abs(row['jacobian_polarization_residual']) <= METRIC_ATOL*(1+abs(row['jacobian_risk_change']))
    with (output/'recomputed-panel-metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt=dict(state='PASS',checked_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
                 protocol='Fixed Stage23 native float32 probe means; all six maps folded into B SiLU(Ah), with external FiLM intact.',
                 protocol_sha256=PROTOCOL_SHA,
                 matrix_contract='PASS',source_files_verified=595,panels=15,CSV_rows=18,means=mean_records,
                 maximum_differences=maximums,tolerances=dict(field=F_TOL,jacobian=J_TOL,metric_atol=METRIC_ATOL,metric_rtol=METRIC_RTOL),
                 input_sha256=hashes,scope='Independent finite-intervention identity and risk arithmetic only. No training, new data, sampling, primary validation, holdout or retrained-function-class claim.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'state':'PASS','panels':15,'rows':18,'maximum_differences':maximums},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    args=parser.parse_args();main(args.runtime,args.analysis)
