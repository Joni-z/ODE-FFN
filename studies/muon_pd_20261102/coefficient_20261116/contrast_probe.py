"""Same-function, fresh-state native AdamW probes in the two stage coordinates."""
import argparse
import copy
import json
from pathlib import Path
from contrast_trial import base, torch, MeanPreservingStageLayer
from sharing_trial import StageOperatorLayer
from contrast_evaluate import evaluator


def weights(layer) -> torch.Tensor:
    if isinstance(layer, MeanPreservingStageLayer):
        return layer.stage_weights()
    return torch.cat([layer.operator()[None], layer.operator()[None]+layer.stage_delta])


def encode(source, mode: str):
    target = copy.deepcopy(source)
    with torch.no_grad():
        for (_, old), (_, new) in zip(source.ode_layers(), target.ode_layers()):
            w = weights(old)
            if mode == 'anchored':
                new.__class__ = StageOperatorLayer
                new.proj.weight.copy_(w[0])
                new.stage_delta.copy_(w[1:]-w[0])
            else:
                new.__class__ = MeanPreservingStageLayer
                c = w.new_tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/2
                new.proj.weight.copy_(w.mean(0))
                new.stage_delta.copy_(torch.einsum('sk,sij->kij', c, w))
            assert torch.allclose(weights(new), w, atol=1e-12, rtol=1e-10)
    return target


def rank_stats(value: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(value.detach())
    return dict(frobenius=float(value.detach().norm()),
                stable_rank=float(s.square().sum()/s[0].square()) if float(s[0]) > 0 else 0.)


def step(model, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mode: str) -> tuple:
    before, raw_p, mean_w = {}, {}, {}
    for name, layer in model.ode_layers():
        before[name] = weights(layer).detach().clone()
        raw_p[name] = layer.operator().detach().clone()
        mean_w[name] = before[name].mean(0)
    prediction = model(x, t)
    loss = (prediction-y).square().mean()
    loss.backward()
    physical, common_energy = {}, {}
    for name, layer in model.ode_layers():
        gp, gb = layer.proj.weight.grad, layer.stage_delta.grad
        if mode == 'anchored':
            g = torch.cat([(gp-gb.sum(0))[None], gb])
        else:
            c = gp.new_tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/2
            g = gp[None]/4+torch.einsum('sk,kij->sij', c, gb)
        physical[name] = g.detach().clone()
        common_energy[name] = float(4*g.mean(0).square().sum()/g.square().sum().clamp_min(1e-30))
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    base.ProjectionOptimizer(model, 'adamw', .0003).step()
    with torch.no_grad():
        after = model(x, t)
        layers = []
        for name, layer in model.ode_layers():
            dp = layer.operator()-raw_p[name]
            dm = weights(layer).mean(0)-mean_w[name]
            denominator = float(dp.norm())
            layers.append(dict(layer=name, P_update=rank_stats(dp), mean_update=rank_stats(dm),
                mean_step_amplification=float(dm.norm())/denominator if denominator > 1e-12 else None,
                gradient_common_energy_fraction=common_energy[name],
                mean_coordinate_residual=float((weights(layer).mean(0)-layer.operator()).norm())))
    return dict(coordinates=mode, pair_mse_before=float(loss.detach()),
        pair_mse_after=float((after-y).square().mean()), grad_norm_before_clip=norm,
        clip_factor=min(1., 1/(norm+1e-6)), field_step_rms=float((after-prediction.detach()).square().mean().sqrt()),
        layers=layers), prediction.detach(), physical


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--reference-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    study, out = Path(a.study), Path(a.output)
    assert not out.exists()
    anchors = json.loads((study/'anchors.json').read_text())
    assert base.digest(Path(anchors['assets'])/'manifest.json') == anchors['assets_manifest_sha256']
    cases = [r for r in anchors['references'] if r['family'] == 'stage_operator_midpoint4']
    if not a.reference_only:
        cases += anchors['new_cases']
    rows = []
    for row in cases:
        folder = Path(row['folder'])
        assert (folder/'ADAPTER_COMPLETE').exists()
        for name, digest in row.get('bound_files', {}).items():
            assert base.digest(folder/name) == digest
        cfg = json.loads((folder/'config.json').read_text())
        cache = base.verify_data(a.data_root, cfg['cell'])
        model, identity = evaluator.ev.restore(folder, cfg, cache/'manifest.json', 20000, 'cpu')
        model.double()
        snapshot = evaluator.ev.tensor_snapshot(model)
        _, fixed, binding = evaluator.assets(anchors['assets'], row['dataset'], 'selection', 'cpu')
        fixed = {k: v.double() for k, v in fixed.items()}
        batches = []
        for offset in range(0, 4096, 512):
            sl = slice(offset, offset+512)
            x, t, y = [fixed[k][sl] for k in ['x', 't', 'y']]
            left, lp, lg = step(encode(model, 'anchored'), x, t, y, 'anchored')
            right, rp, rg = step(encode(model, 'contrast'), x, t, y, 'contrast')
            function_error = float((lp-rp).abs().max())
            gradient_error = max(float((lg[n]-rg[n]).abs().max()) for n in lg)
            assert torch.allclose(lp, rp, atol=1e-11, rtol=1e-10)
            assert all(torch.allclose(lg[n], rg[n], atol=1e-11, rtol=1e-10) for n in lg)
            batches.append(dict(batch=offset//512, function_error=function_error,
                physical_gradient_error=gradient_error, anchored=left, contrast=right))
        evaluator.ev.same_tensors(model, snapshot)
        rows.append(dict(id=row['id'], dataset=row['dataset'], checkpoint=identity,
            assets_sha256=binding['sha256'], parameters_unchanged=True, batches=batches))
        base.dump(out.with_name(out.name+'-progress.json'), dict(completed=len(rows), total=len(cases)))
    base.dump(out, dict(passed=True, reference_only=a.reference_only,
        scope='same frozen functions; fresh native optimizer states, not historical momentum replay',
        rows=rows, anchors_sha256=base.digest(study/'anchors.json'), source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True, cases=len(rows), output=str(out))))


if __name__ == '__main__':
    main()
