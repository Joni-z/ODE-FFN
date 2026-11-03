"""Independent NumPy replay of saved cycle14 arrays; no model imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAG = 'professor_validation14_20261104'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape, x.dtype.str)).encode() + x.tobytes()).hexdigest()


def sw2(x, y, directions):
    if not np.isfinite(x).all():
        return None
    u = np.sort(x.astype(np.float64) @ directions.astype(np.float64).T, axis=0)
    v = np.sort(y.astype(np.float64) @ directions.astype(np.float64).T, axis=0)
    return float(np.sqrt(np.mean(np.square(u-v))))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return dict(n=len(values), finite=len(finite),
                mean=float(np.mean(finite)) if len(finite) == len(values) else None,
                sd=float(np.std(finite, ddof=1)) if len(finite) == len(values) else None,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None)


def label(case):
    name = {'pd_isru_k1': 'ISRU-D', 'free_isru': 'Free ISRU',
            'swiglu64': 'Swi64', 'swiglu192': 'Swi192'}[case['family']]
    return f"{name} / {case['optimizer']} {case['projection_lr']:g}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--host', action='append', choices=['145', 'lenovo'], help='Explicit partial-host audit; missing cohort is reported.')
    a = p.parse_args()
    runtime, out = Path(a.runtime_root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    grid = json.loads((HERE/'grid.json').read_text())
    cases = [c for c in grid['cells'] if not a.host or c['host'] in a.host]
    rows, fields, anchors, files, common = [], [], [], [], {}
    max_sw2_error = 0.
    for case in cases:
        folder = runtime/case['host']/'output'/TAG/case['dataset']/case['id']/'seed000'
        assert (folder/'ADAPTER_COMPLETE').exists()
        result = json.loads((folder/'result.json').read_text())
        binding = json.loads((folder/'binding.json').read_text())
        assert binding['case'] == case and binding['model_seed'] == 0
        assert binding['grid_sha256'] == digest(HERE/'grid.json')
        assert binding['source_sha256'] == digest(HERE/'audit.py')
        assert result['passed'] and result['evaluations'] == 48 and result['tensors_unchanged']
        with np.load(folder/'banks.npz', allow_pickle=False) as saved:
            banks = {k: saved[k].copy() for k in saved.files}
        hashes = {k: array_hash(v) for k, v in banks.items()}
        assert hashes == binding['bank_identity'] == result['bank_identity']
        if case['dataset'] in common:
            assert hashes == common[case['dataset']], 'models did not share sampling arrays'
        else:
            common[case['dataset']] = hashes
        with np.load(folder/'native_anchor.npz', allow_pickle=False) as z:
            native_score = sw2(z['generated'], z['target'], z['directions'])
            archive_score = sw2(z['archived_generated'], z['target'], z['directions'])
            exact = np.array_equal(z['generated'], z['archived_generated'])
            coordinate_error = float(np.max(np.abs(z['generated']-z['archived_generated'])))
            if grid['execution_devices'][case['host']] == 'cuda':
                assert exact
            else:
                assert np.allclose(z['generated'], z['archived_generated'], rtol=1e-5, atol=1e-6)
        anchor = json.loads((folder/'native_anchor.json').read_text())
        assert exact == anchor['generated_exact']
        assert coordinate_error == anchor['generated_max_abs_error']
        assert abs(native_score-case['native_test_sw2']) < 2e-6
        assert abs(archive_score-case['native_test_sw2']) < 1e-10
        anchors.append(dict(cell=case['id'], dataset=case['dataset'], host=case['host'],
            device=result['device'], label=label(case), native_archive_exact=exact,
            max_coordinate_error=coordinate_error, replayed_sw2=native_score,
            original_archive_sw2=archive_score, source_sw2=case['native_test_sw2']))
        for field in result['fields']:
            step = field['step']
            with np.load(folder/f'field_{step}.npz', allow_pickle=False) as z:
                mse = float(np.mean((z['prediction'].astype(float)-z['oracle'].astype(float))**2))
            assert abs(mse-field['oracle_mse']) < 1e-12
            if step == 20000:
                assert abs(mse-case['native_field_mse']) < 2e-6
            fields.append(dict(cell=case['id'], dataset=case['dataset'], label=label(case),
                               checkpoint_step=step, oracle_mse=mse, **field))
        records = [json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
        assert len(records) == len({(r['checkpoint_step'], r['bank'], r['heun_steps']) for r in records}) == 48
        for row in records:
            assert row['cell'] == case['id'] and row['model_seed'] == 0
            assert row['checkpoint_step'] in grid['checkpoint_steps']
            assert row['bank'] in range(8) and row['heun_steps'] in grid['heun_steps']
            assert row['evaluation_stream'] == grid['evaluation_stream_base'] + row['bank']
            cp_key = next(k for k in case['file_hashes'] if k.replace('\\', '/').endswith(f"step_{row['checkpoint_step']:07d}.pt"))
            assert row['checkpoint_sha256'] == case['file_hashes'][cp_key]
            path = folder/row['generated_file']
            assert digest(path) == row['generated_sha256']
            generated = np.load(path, allow_pickle=False)
            assert generated.shape == (4096, banks['directions'].shape[1])
            bank = row['bank']
            score = sw2(generated, banks[f'target_{bank}'], banks['directions'])
            floor = sw2(banks[f'target_{bank}'], banks[f'target2_{bank}'], banks['directions'])
            assert abs(floor-row['target_target_sw2']) < 1e-10
            if score is None:
                assert row['sw2'] is None
            else:
                error = abs(score-row['sw2'])
                assert error < 1e-10
                max_sw2_error = max(max_sw2_error, error)
            assert float(np.isfinite(generated).all(axis=1).mean()) == row['finite_fraction']
            rows.append(dict(row, numpy_sw2=score, numpy_floor=floor, label=label(case), device=result['device']))
            files.append(dict(file=str(path.relative_to(runtime)), sha256=digest(path)))
        print(f"Replayed {case['id']}: {len(rows)}/{48*len(cases)}", flush=True)
    assert len(rows) == 48*len(cases) and len(fields) == 3*len(cases) and len(anchors) == len(cases)
    groups = []
    for case in cases:
        for step in grid['checkpoint_steps']:
            for heun in grid['heun_steps']:
                selected = [r for r in rows if r['cell'] == case['id'] and r['checkpoint_step'] == step and r['heun_steps'] == heun]
                assert len(selected) == 8
                groups.append(dict(cell=case['id'], dataset=case['dataset'], label=label(case),
                    checkpoint_step=step, heun_steps=heun, **summary([r['numpy_sw2'] for r in selected]),
                    floor_mean=float(np.mean([r['numpy_floor'] for r in selected]))))
    pairs = []
    for candidate in [c for c in cases if c['family'] == 'pd_isru_k1']:
        for reference in [c for c in cases if c['dataset'] == candidate['dataset'] and c['id'] != candidate['id']]:
            for step in grid['checkpoint_steps']:
                for heun in grid['heun_steps']:
                    def bank_values(cell):
                        return {r['bank']: r['numpy_sw2'] for r in rows if r['cell'] == cell['id'] and r['checkpoint_step'] == step and r['heun_steps'] == heun}
                    c, r = bank_values(candidate), bank_values(reference)
                    assert c.keys() == r.keys() == set(range(8))
                    delta = [None if c[b] is None or r[b] is None else c[b]-r[b] for b in range(8)]
                    pairs.append(dict(dataset=candidate['dataset'], candidate=label(candidate), reference=label(reference),
                        checkpoint_step=step, heun_steps=heun, **summary(delta),
                        candidate_wins=sum(v is not None and v < 0 for v in delta)))
    write_csv(out/'recomputed.csv', rows)
    write_csv(out/'field_trajectories.csv', fields)
    write_csv(out/'native_anchors.csv', anchors)
    write_csv(out/'sampling_summary.csv', groups)
    write_csv(out/'paired_differences.csv', pairs)
    receipt = dict(passed=True, model_seed=0, training_runs=0, formal_evaluations=len(rows),
                   complete_cohort=len(cases)==10, hosts=sorted({c['host'] for c in cases}),
                   pending_cells=[c['id'] for c in grid['cells'] if c not in cases],
                   physical_checkpoints=len(fields), native_anchors=len(anchors), common_arrays_verified=True,
                   numpy_sw2_max_error=max_sw2_error, all_finite=all(r['finite_fraction'] == 1 for r in rows),
                   grid_sha256=digest(HERE/'grid.json'), producer_sha256=digest(HERE/'audit.py'),
                   analyzer_sha256=digest(__file__), generated_arrays=files,
                   interpretation='Evaluation banks condition on fixed seed0 fits; no training replication.')
    (out/'replay.json').write_text(json.dumps(receipt, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for dataset in {c['dataset'] for c in cases}:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, heun in zip(axes, [32, 128]):
            for case in [c for c in cases if c['dataset'] == dataset]:
                selected = sorted([r for r in groups if r['cell'] == case['id'] and r['heun_steps'] == heun], key=lambda x:x['checkpoint_step'])
                x = [r['checkpoint_step']/1000 for r in selected]
                mean = np.array([r['mean'] for r in selected], dtype=float)
                sd = np.array([r['sd'] for r in selected], dtype=float)
                line, = ax.plot(x, mean, 'o-', label=label(case), linewidth=1.5)
                ax.fill_between(x, mean-sd, mean+sd, color=line.get_color(), alpha=.12)
            ax.set(xlabel='Checkpoint (thousand updates)', ylabel='Sliced W2 (lower is better)', title=f'{dataset}: Heun {heun}', xticks=[10,15,20])
        axes[1].legend(fontsize=8)
        fig.savefig(out/f'{dataset}-sampling.png', dpi=180)
        plt.close(fig)
    print(json.dumps({k:v for k,v in receipt.items() if k!='generated_arrays'}))


if __name__ == '__main__':
    main()
