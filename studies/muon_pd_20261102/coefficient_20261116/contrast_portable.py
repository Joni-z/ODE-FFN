"""Export two trained controls for the existing sampling/density observers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from contrast_trial import FAMILY, build_model, base, torch, HERE
import contrast_evaluate as evaluation
from gpu_resolution import load

REPO = HERE.parents[2]


def copy_bound(source: Path, root: Path, relative: str) -> tuple[str, str]:
    target = root/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert base.digest(source) == base.digest(target)
    return relative, base.digest(target)


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['study', 'density-reference', 'data-root', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    study, previous, out = map(Path, [a.study, a.density_reference, a.output])
    assert json.loads((study/'analysis/COMPLETE.json').read_text())['passed']
    anchors = json.loads((study/'anchors.json').read_text())
    old_manifest = json.loads((previous/'portable.json').read_text())
    assert anchors['passed'] and old_manifest['passed'] and len(anchors['new_cases']) == 2
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    # All imported project-local Python dependencies are bound, including the
    # inherited builder chain. Native vendored sources are also checked above.
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(REPO/'studies/muon_pd_20261102'):
                sources[str(path.relative_to(REPO))] = base.digest(path)
    for name in ['gpu_resolution.py', 'contrast_trial.py']:
        assert str((HERE/name).relative_to(REPO)) in sources
    manifests = {'sampling-models': [], 'density-models': []}
    checks = []
    for item in anchors['new_cases']:
        folder, id = Path(item['folder']), item['id']
        cfg = json.loads((folder/'config.json').read_text())
        assert cfg['cell']['family'] == FAMILY
        cache = base.verify_data(a.data_root, cfg['cell'])
        native, identity = evaluation.evaluator.ev.restore(folder, cfg, cache/'manifest.json', 20000, 'cpu')
        assert len(list(native.ode_layers())) == 6
        arrays, fixed, binding = evaluation.evaluator.assets(anchors['assets'], item['dataset'], 'confirmation', 'cpu')
        evaluated = study/'analysis/confirmation'/id
        pred, _ = evaluation.evaluator.ev.field(native, fixed)
        assert np.array_equal(pred, np.load(evaluated/'field-20000.npy'))
        value = base.train.generated_samples(native, torch.from_numpy(arrays['source_0']), 128).numpy()
        assert np.array_equal(value, np.load(evaluated/'20000-b0.npy'))
        for mode, cases in manifests.items():
            root = out/mode
            case = dict(id=id, dataset=item['dataset'], family=FAMILY, builder_sources_sha256=sources.copy())
            for key, file in [('config', folder/'config.json'), ('checkpoint', folder/'checkpoints/step_0020000.pt'),
                              ('dataset_manifest', cache/'manifest.json')]:
                case[key], case[key+'_sha256'] = copy_bound(file, root, f'cases/{id}/{file.name}')
            if mode == 'sampling-models':
                case['assets'], case['assets_sha256'] = copy_bound(Path(binding['file']), root, f"assets/{item['dataset']}.npz")
                assert case['assets_sha256'] == binding['sha256']
                case['cpu_field'], case['cpu_field_sha256'] = copy_bound(evaluated/'field-20000.npy', root, f'cases/{id}/cpu_field.npy')
                pairs = [copy_bound(evaluated/f'20000-b{i}.npy', root, f'cases/{id}/cpu-b{i}.npy') for i in range(8)]
                case['cpu_samples'], case['cpu_sample_sha256'] = [v[0] for v in pairs], [v[1] for v in pairs]
                summary = json.loads((evaluated/'summary.json').read_text())
                case['original_cpu_pooled_sw2'] = next(x['pooled_sw2'] for x in summary if x['step'] == 20000)
            else:
                original = next(c for c in old_manifest['cases'] if c['dataset'] == item['dataset'])
                assert base.digest(previous/original['assets']) == original['assets_sha256']
                case['assets'], case['assets_sha256'] = copy_bound(previous/original['assets'], root, f"assets/{item['dataset']}.npz")
            # Actual portable restoration must match the real trained evaluator,
            # including nonpersistent Fourier buffers and the contrast semantics.
            portable = load(root, case, 'cpu')
            with torch.no_grad():
                assert torch.equal(portable(fixed['x'][:64], fixed['t'][:64]), native(fixed['x'][:64], fixed['t'][:64]))
            cases.append(case)
        checks.append(dict(id=id, native_full_field_exact=True, native_bank0_exact=True,
            portable_restore_exact=True, checkpoint_sha256=identity['checkpoint_sha256']))
        base.dump(out/'progress.json', dict(completed=len(checks), checks=checks))
    for mode, cases in manifests.items():
        base.dump(out/mode/'portable.json', dict(passed=True, cases=cases,
            source_sha256=base.digest(__file__), source_study_completion_sha256=base.digest(study/'analysis/COMPLETE.json'),
            original_density_portable_sha256=base.digest(previous/'portable.json') if mode == 'density-models' else None))
    base.dump(out/'COMPLETE.json', dict(passed=True, cases=2, modes=list(manifests), checks=checks,
        source_sha256=base.digest(__file__), builder_source_files=len(sources)))
    print(json.dumps(dict(passed=True, cases=2, output=str(out))))


if __name__ == '__main__':
    main()
