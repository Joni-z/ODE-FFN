"""Complete same-bank comparison; no model execution or partial-cohort report."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from analyze_transport import compositions, interval, sha
from verify_ordinary_population import statistics


class PendingCohort(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sign_probability(delta):
    nonzero = delta[delta != 0]
    n = len(nonzero)
    if n == 0:
        return 1.
    smaller = min(int(np.count_nonzero(nonzero < 0)), int(np.count_nonzero(nonzero > 0)))
    return min(1., 2*sum(math.comb(n, k) for k in range(smaller+1))/2**n)


def holm(values):
    order = np.argsort(values, kind='stable')
    adjusted, previous = np.empty(len(values)), 0.
    for index, original in enumerate(order):
        previous = max(previous, min(1., (len(values)-index)*values[original]))
        adjusted[original] = previous
    return adjusted


def validate_ordinary(root, binding):
    terminal = root/'TERMINAL.json'
    if not terminal.exists():
        raise PendingCohort('The original ordinary-model controller has not written TERMINAL.json')
    done = read(terminal)
    assert done['passed'] and len(done['completed']) == 8 and not done['unstarted']
    assert all(row['actual_terminal_exit'] == 0 for row in done['completed'])
    assert {(row['index'], row['stage']) for row in done['completed']} == {
        (i, stage) for i in range(4) for stage in ('evaluate', 'verify')}
    result = []
    for entry in binding['cases']:
        folder = root/f"index{entry['index']}"
        complete, independent = read(folder/'COMPLETE.json'), read(folder/'independent.json')
        assert complete['passed'] and independent['passed']
        assert not complete['witness'] and not independent['witness']
        assert independent['complete_sha256'] == sha(folder/'COMPLETE.json')
        assert independent['rows_sha256'] == sha(folder/'rows.json')
        assert complete['case'] == entry['case'] and complete['parameters_buffers_rng_unchanged']
        rows = read(folder/'rows.json')
        assert len(rows) == 4 and {(r['split'], r['steps']) for r in rows} == {
            (split, steps) for split in ('selection', 'confirmation') for steps in (128, 2048)}
        for row in rows:
            assert row['points_per_bank'] == 4096 and row['banks'] == 8
            assert sha(folder/row['file']) == row['sha256']
        result.append((entry, complete, folder, rows))
    return result


def gather(a):
    binding = read(a.binding)
    assert binding['passed'] and binding['training_seed'] == 0 and len(binding['cases']) == 4
    ordinary = validate_ordinary(a.input, binding)
    portable_root = Path(binding['population_root'])
    assert sha(portable_root/'portable.json') == binding['population_manifest_sha256']
    portable = read(portable_root/'portable.json')
    assert len(portable['cases']) == 4 and portable['nonpersistent_buffers_included']
    ode_terminal = read(a.ode_input/'TERMINAL.json')
    assert ode_terminal['passed'] and len(ode_terminal['cases']) == 8
    assert all(r['actual_terminal_exit'] == 0 for r in ode_terminal['cases'])
    cost = read(a.costs)
    assert sha(a.costs.parents[3]/cost['source_file']) == cost['source_sha256']
    cost_by_id = {r['id']: r for r in cost['rows']}
    rows, bindings, specs = [], {str(a.binding): sha(a.binding), str(a.costs): sha(a.costs)}, {}
    bindings[str(portable_root/'portable.json')] = sha(portable_root/'portable.json')
    cost_source = a.costs.parents[3]/cost['source_file']
    bindings[str(cost_source)] = sha(cost_source)
    bindings[str(a.input/'TERMINAL.json')] = sha(a.input/'TERMINAL.json')
    bindings[str(a.ode_input/'TERMINAL.json')] = sha(a.ode_input/'TERMINAL.json')
    for entry, complete, folder, data in ordinary:
        case = entry['case']; dataset = case['dataset']
        assert complete['binding_sha256'] == sha(a.binding)
        costs = cost_by_id[case['id']]
        assert costs['checkpoint_sha256'] == case['checkpoint_sha256']
        assert costs['parameters'] == complete['parameters']
        law_file = Path(binding['control_root'])/case['dataset_manifest']
        assert sha(law_file) == case['dataset_manifest_sha256']
        spec = read(law_file)['datasets'][dataset]
        if dataset in specs:
            assert specs[dataset] == (spec, case['dataset_manifest_sha256'])
        specs[dataset] = spec, case['dataset_manifest_sha256']
        for name in ('COMPLETE.json', 'independent.json', 'rows.json'):
            bindings[str(folder/name)] = sha(folder/name)
        for item in data:
            references = [r for r in portable['cases'] if r['dataset'] == dataset]
            assert len(references) == 2
            source = entry['splits'][item['split']]
            assert all(r['splits'][item['split']]['assets']['sha256'] == source['population_sha256']
                       for r in references)
            with np.load(folder/item['file']) as saved:
                samples = saved['samples'].copy()
            bindings[str(folder/item['file'])] = sha(folder/item['file'])
            label = 'Free ISRU' if case['family'] == 'free_isru' else (
                'SwiGLU64 / Muon' if dataset == 'gmm_ring8_2d' else 'SwiGLU192 / AdamW')
            rows.append(dict(dataset=dataset, model=case['id'], label=label, ordinary=True,
                family=case['family'], split=item['split'], steps=item['steps'],
                parameters=costs['parameters'], dense_macs=costs['dense_macs'],
                pooled_sw2=item['pooled_sw2'], bank_sw2=item['bank_sw2'],
                sample_stats=item['sample_stats'], checkpoint_sha256=case['checkpoint_sha256']))
    for case in portable['cases']:
        folder = a.ode_input/case['id']
        done, independent = read(folder/'COMPLETE.json'), read(folder/'independent.json')
        assert done['passed'] and independent['passed'] and not done['witness']
        assert done['parameters_buffers_rng_unchanged']
        assert done['portable_sha256'] == binding['population_manifest_sha256']
        assert independent['complete_sha256'] == sha(folder/'COMPLETE.json')
        assert independent['rows_sha256'] == sha(folder/'rows.json')
        spec, manifest_hash = specs[case['dataset']]
        assert case['identity']['manifest_sha256'] == manifest_hash
        tensor_file = portable_root/case['tensor_file']
        assert sha(tensor_file) == case['tensor_sha256']
        bindings[str(tensor_file)] = sha(tensor_file)
        with np.load(tensor_file) as tensors:
            parameters = {k: v for k, v in tensors.items() if k.startswith('parameter::')}
        operators = [v for k, v in parameters.items() if '.ode' in k and k.endswith('proj.weight')]
        assert len(operators) == 6
        count = sum(v.size for v in parameters.values())
        macs = sum(v.size for v in parameters.values() if v.ndim == 2)+7*sum(v.size for v in operators)
        assert macs == (227648 if spec['dim'] == 2 else 229440)
        for name in ('COMPLETE.json', 'independent.json', 'rows.json'):
            bindings[str(folder/name)] = sha(folder/name)
        selected = [r for r in read(folder/'rows.json') if r['steps'] in (128, 2048)]
        assert len(selected) == 4
        for item in selected:
            assert item['points_per_bank'] == 4096 and sha(folder/item['file']) == item['sha256']
            bindings[str(folder/item['file'])] = sha(folder/item['file'])
            with np.load(folder/item['file']) as saved:
                for kind in ('native', 'crossed'):
                    samples = saved[kind].copy()
                    assert samples.shape == (8, 4096, spec['dim']) and np.isfinite(samples).all()
                    inference = case['trained_solver'] if kind == 'native' else case['inference_solver']
                    label = ('M' if case['trained_solver'] == 'midpoint' else 'R')+'→'+(
                        'M' if inference == 'midpoint' else 'R')
                    rows.append(dict(dataset=case['dataset'], model=case['id']+'::'+kind,
                        label=label, ordinary=False, family='state_flow', split=item['split'],
                        steps=item['steps'], parameters=count, dense_macs=macs,
                        pooled_sw2=item[kind]['pooled_sw2'], bank_sw2=item[kind]['bank_sw2'],
                        sample_stats=statistics(np.concatenate(samples), spec),
                        checkpoint_sha256=case['identity']['checkpoint_sha256']))
    assert len(rows) == 48
    assert len({(r['model'], r['split'], r['steps']) for r in rows}) == 48
    return rows, bindings


def contrasts(rows):
    counts, weights = compositions()
    comparisons = []
    for ode in (r for r in rows if not r['ordinary'] and r['steps'] == 2048):
        for control in (r for r in rows if r['ordinary'] and
                        (r['dataset'], r['split'], r['steps']) ==
                        (ode['dataset'], ode['split'], ode['steps'])):
            delta = np.asarray(ode['bank_sw2'])**2-np.asarray(control['bank_sw2'])**2
            distribution = counts@delta/8
            comparisons.append(dict(dataset=ode['dataset'], split=ode['split'],
                ode_model=ode['model'], ode_label=ode['label'], control_model=control['model'],
                control_label=control['label'], steps=2048, bank_squared_loss_difference=delta.tolist(),
                pooled_squared_loss_difference=ode['pooled_sw2']**2-control['pooled_sw2']**2,
                mean=float(delta.mean()), median=float(np.median(delta)), sd=float(delta.std(ddof=1)),
                banks_improving=int(np.count_nonzero(delta < 0)),
                ci95=interval(distribution, weights, .025),
                ci_adjusted=interval(distribution, weights, .05/64),
                exact_sign_p=sign_probability(delta),
                ode_to_control_dense_work=ode['dense_macs']/control['dense_macs'],
                ode_sample_stats=ode['sample_stats'], control_sample_stats=control['sample_stats']))
    assert len(comparisons) == 32
    corrected = holm([c['exact_sign_p'] for c in comparisons])
    for row, value in zip(comparisons, corrected, strict=True):
        row['holm_sign_p'] = float(value)
    return comparisons


def write(a, rows, bindings, comparisons):
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    figures = a.output/'figures'; figures.mkdir()
    summary = dict(rows=rows, contrasts=comparisons, source_bindings=bindings,
        training_seed=0, paired_banks=8, points_per_bank=4096, pooled_points=32768,
        primary_steps=2048, primary_contrasts=32, paired_bootstrap_multisets=6435,
        paired_bootstrap_ordered_samples=8**8, adjusted_tail=.05/64,
        source_sha256=sha(Path(__file__)), statistics_source_sha256=sha(Path(__file__).with_name('analyze_transport.py')))
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    table = ['| Task | Stream | Model | SW2 128 | SW2 2048 | Parameters | Dense MACs |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in sorted((r for r in rows if r['steps'] == 2048),
                      key=lambda r: (r['dataset'], r['split'], r['ordinary'], r['label'])):
        coarse = next(r for r in rows if (r['model'], r['split'], r['steps']) ==
                      (row['model'], row['split'], 128))
        task = 'Ring8' if row['dataset'] == 'gmm_ring8_2d' else 'Product16'
        table.append(f"| {task} | {row['split']} | {row['label']} | {coarse['pooled_sw2']:.3f} | "
                     f"{row['pooled_sw2']:.3f} | {row['parameters']:,} | {row['dense_macs']:,} |")
    (a.output/'table.md').write_text('\n'.join(table)+'\n')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    tasks = [('gmm_ring8_2d', 'Ring8'), ('rotated_product_bimodal_16d', 'Product16')]
    order = ['M→M', 'M→R', 'R→R', 'R→M', 'Free ISRU', 'SwiGLU']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (dataset, name) in enumerate(tasks):
        for j, split in enumerate(('selection', 'confirmation')):
            axis = axes[i, j]
            for steps, color, marker in [(128, '#0072B2', 'o'), (2048, '#D55E00', 's')]:
                values = []
                for label in order:
                    selected = [r for r in rows if r['dataset'] == dataset and r['split'] == split
                        and r['steps'] == steps and (r['label'].startswith('SwiGLU') if label == 'SwiGLU' else r['label'] == label)]
                    assert len(selected) == 1
                    values.append(selected[0]['pooled_sw2'])
                axis.scatter(np.arange(6)+(-.08 if steps == 128 else .08), values,
                             color=color, marker=marker, label=f'Heun{steps}')
            axis.set_xticks(range(6), order, rotation=15)
            axis.set_title(name+' / '+split); axis.set_ylabel('Pooled SW2 (lower is better)')
            axis.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(figures/('same-bank-quality.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    labels = [label+' vs '+control for label in order[:4] for control in ('Free ISRU', 'SwiGLU')]
    for axis, (dataset, name) in zip(axes, tasks, strict=True):
        for index, label in enumerate(labels):
            ode_label, control_label = label.split(' vs ')
            for split, offset, color in [('selection', -.13, '#0072B2'), ('confirmation', .13, '#D55E00')]:
                row = next(c for c in comparisons if c['dataset'] == dataset and c['split'] == split
                    and c['ode_label'] == ode_label and c['control_label'].startswith(control_label))
                y = index+offset
                axis.scatter(row['bank_squared_loss_difference'], y+np.linspace(-.035, .035, 8),
                             color=color, alpha=.45, s=10)
                axis.plot(row['ci_adjusted'], [y, y], color=color, linewidth=2)
                axis.scatter([row['mean']], [y], color=color, marker='D', s=22,
                             label=split if index == 0 else None)
        axis.axvline(0, color='.5', linewidth=.8); axis.set_yticks(range(8), labels)
        axis.invert_yaxis(); axis.set_title(name+' / Heun2048')
        axis.set_xlabel('ODE minus ordinary bank squared SW2')
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0)); axis.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(figures/('paired-ordinary-contrasts.'+ext), dpi=180)
    plt.close(fig)
    overview = []
    for dataset, name in tasks:
        selected = [c for c in comparisons if c['dataset'] == dataset]
        overview.append(f"{name}: ODE has lower pooled SW2 in {sum(c['pooled_squared_loss_difference'] < 0 for c in selected)}/16 "
                        f"comparisons; {sum(c['ci_adjusted'][1] < 0 for c in selected)}/16 have a negative adjusted upper bootstrap bound.")
    (a.output/'analysis-report.md').write_text('# Same-bank ordinary controls\n\n'+'\n\n'.join(overview)+
        '\n\nAll original models use the same banks, targets, directions and native sampler. M/R denote midpoint/RK4 training and inference. '
        'Ordinary optimizer and capacity labels remain fixed. Every producer and independent metric check completed before this report.\n\n'
        '![Common-bank quality](figures/same-bank-quality.png)\n\n'
        'The two meshes are prescribed observations. Points are complete pooled scores, without error bars. '
        'The comparison determines whether the useful internal-solver effect reaches ordinary-model quality.\n\n'+
        '\n'.join(table)+'\n\n![Paired effects](figures/paired-ordinary-contrasts.png)\n\n'
        'Each contrast retains eight paired banks; negative favors ODE. Diamonds show means, lines the registered adjusted bootstrap intervals. '
        'These conditional intervals and the separate exact sign tests measure different summaries. Full values are in summary.json.\n\n'
        '## Claim candidates\n\n- Claim: the listed same-bank quality and matrix-work comparisons.\n'
        '  - Evidence: all 48 model/stream/mesh rows and 32 primary contrasts, bound to complete independent outputs.\n'
        '  - Allowed wording: task- and configuration-specific comparison at the recorded budgets.\n'
        '  - Uncertainty: fixed trained weights and finite evaluation banks; different ordinary configurations remain labeled.\n'
        '  - Next check: review mode/gap effects and the independent comparison audit before any image decision.\n'
        '  - Decision: retain the complete comparison; no automatic image promotion.\n')
    (a.output/'stats-appendix.md').write_text('# Statistical scope\n\n'
        'Eight independent paired source/target banks at fixed weights; 4,096 samples each. '
        'Report all bank squared-distance differences, means, medians and sample SDs. '
        'The registered 32 primary contrasts use exhaustive multinomial bootstrap enumeration: '
        '6,435 count vectors, multiplicities totaling 8^8, percentile tail 0.05/64. '
        'These are conditional intervals without guaranteed coverage. They are not p-values. '
        'The secondary two-sided exact binomial sign test excludes zero differences and tests balanced sign probabilities; '
        'Holm correction covers the same 32 contrasts. Pooled SW2 is not the mean bank SW2. '
        'Dense MACs exclude pointwise operations; CPU and GPU runtimes are not compared. '
        'Formal model training used seed 0 throughout; bank draws do not represent repeated training.\n')
    (a.output/'figure-catalog.md').write_text('# Figure purposes\n\n'
        '- same-bank-quality: every model on both streams and both prescribed meshes. Determines practical ordering without choosing a mesh. '
        'No error bars: the plotted values are empirical pooled distances.\n'
        '- paired-ordinary-contrasts: all 32 primary contrasts, raw bank effects, means and adjusted conditional intervals. '
        'Checks whether pooled ordering also has paired-bank support. Review both tasks, including contrary effects.\n')


def check_statistics():
    assert sign_probability(np.zeros(8)) == 1
    assert sign_probability(-np.ones(8)) == .0078125
    assert sign_probability(np.array([-1.]*7+[1.])) == .0703125
    assert sign_probability(np.array([-1.]*4+[1.]*4)) == 1
    assert np.array_equal(holm([.0078125]*32), np.full(32, .25))
    counts, weights = compositions()
    assert np.allclose(interval(counts@np.full(8, -.3)/8, weights, .05/64), [-.3, -.3], atol=1e-15, rtol=0)
    return dict(passed=True, exact_sign_and_holm=True, constant_bootstrap=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('binding', 'input', 'ode-input', 'costs', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--check-statistics', action='store_true')
    parser.add_argument('--wait-for-pid', type=int)
    a = parser.parse_args()
    if a.check_statistics:
        print(json.dumps(check_statistics())); return
    while True:
        try:
            rows, bindings = gather(a)
            break
        except PendingCohort as error:
            if a.wait_for_pid is None:
                print(json.dumps(dict(status='pending', reason=str(error), artifacts_written=False)))
                return
            command = subprocess.check_output(['ps', '-p', str(a.wait_for_pid), '-o', 'command='], text=True)
            assert 'ordinary_population.py --mode dispatch' in command and str(a.input) in command
            print('WAITING_ON_ORIGINAL_ORDINARY_CONTROLLER', a.wait_for_pid, flush=True)
            time.sleep(30)
    if a.audit_only:
        print(json.dumps(dict(status='complete', model_rows=len(rows), artifacts_written=False)))
        return
    write(a, rows, bindings, contrasts(rows))
    subprocess.run([sys.executable, str(Path(__file__).with_name('verify_ordinary_comparison.py')),
        '--input', str(a.output), '--binding', str(a.binding), '--ordinary-input', str(a.input),
        '--ode-input', str(a.ode_input), '--output', str(a.output/'independent.json')], check=True)
    print('ORDINARY_COMPARISON_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
