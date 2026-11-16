"""Complete matched-schedule numerical comparison from independently audited arrays."""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from contrast_analyze import compositions
from sharing_analyze import read, sha, dump, csvrows, holm

TASKS = ['gmm_ring8_2d', 'rotated_product_bimodal_16d']
PAL, ALT, SHARED = 'palindrome_operator_midpoint4', 'alternating_operator_midpoint4', 'state_midpoint4'
SETTINGS = [(4, 128), (4, 256), (4, 512), (8, 128), (16, 128)]


def intervals(delta: np.ndarray) -> dict:
    counts = np.asarray(list(compositions(8, 8)))
    multiplicity = np.asarray([math.factorial(8)//math.prod(math.factorial(int(v)) for v in row) for row in counts])
    assert len(counts) == 6435 and multiplicity.sum() == 8**8
    values = counts@delta/8
    order = np.argsort(values)
    cdf = np.cumsum(multiplicity[order]/8**8)
    at = lambda q: float(values[order[np.searchsorted(cdf, q)]])
    return dict(marginal95=[at(.025), at(.975)], nominal_ten_contrast_bonferroni=[at(.0025), at(.9975)])


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'shared-refinement', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    study, reference, out = map(Path, [args.study, args.shared_refinement, args.output])
    verified = study/'numerical-verified'
    proof = read(verified/'COMPLETE.json')
    assert proof['passed'] and proof['cases'] == 8 and proof['rows'] == 24 and proof['max_numpy_difference'] < 1e-10
    assert read(study/'numerical-bankwise/queue-finished.json')['passed']
    ref_proof = read(reference/'COMPLETE.json')
    assert ref_proof['passed'] and ref_proof['pooled_rows'] == 16
    manifest = read(study/'portable/sampling-models/portable.json')
    assert sha(study/'portable/sampling-models/portable.json') == proof['portable_manifest_sha256']
    cases = {c['id']: c for c in manifest['cases']}
    raw, banks = csvrows(verified/'scores.csv'), csvrows(verified/'banks.csv')
    assert len(raw) == 24 and len(banks) == 192
    rows, bank_values = {}, {}
    for row in raw:
        case = cases[row['id']]
        assert row['checkpoint_sha256'] == case['checkpoint_sha256']
        key = (row['dataset'], row['family'], int(row['inner']), int(row['outer']))
        values = sorted([b for b in banks if b['id'] == row['id'] and b['mode'] == row['mode'] and b['inner'] == row['inner'] and b['outer'] == row['outer']], key=lambda b: int(b['bank']))
        assert [int(b['bank']) for b in values] == list(range(8))
        value = np.array([float(b['score']) for b in values])
        if key in rows:
            assert row['sample_sha256'] == rows[key]['sample_sha256']
            assert np.array_equal(value, bank_values[key])
            continue
        rows[key] = dict(row, score=float(row['score']), inner=int(row['inner']), outer=int(row['outer']), assets_sha256=case['assets_sha256'])
        bank_values[key] = value
    assert len(rows) == 20
    anchors = read(study/'anchors.json')
    for row in read(reference/'scores.json')['rows']:
        if row['family'] != SHARED:
            continue
        case = next(c for c in cases.values() if c['dataset'] == row['dataset'])
        anchor = next(c for c in anchors['references'] if c['dataset'] == row['dataset'] and c['family'] == SHARED)
        assert row['assets_sha256'] == case['assets_sha256']
        assert row['checkpoint_sha256'] == anchor['bound_files']['checkpoints/step_0020000.pt']
        key = (row['dataset'], SHARED, row['inner'], row['outer'])
        rows[key] = dict(row, score=row['pooled_sw2'])
        bank_values[key] = np.asarray(row['bank_sw2'])
    assert len(rows) == 28
    contrasts = []
    for task in TASKS:
        for inner, outer in SETTINGS:
            left, right = (task, PAL, inner, outer), (task, ALT, inner, outer)
            delta = bank_values[left]-bank_values[right]
            nonzero = delta[delta != 0]
            sd = float(delta.std(ddof=1))
            contrasts.append(dict(dataset=task, inner=inner, outer=outer,
                palindrome=rows[left]['score'], alternating=rows[right]['score'],
                pooled_difference=rows[left]['score']-rows[right]['score'],
                bank_differences=delta.tolist(), paired_mean=float(delta.mean()), paired_sd=sd,
                paired_standardized_effect=float(delta.mean())/sd if sd else None,
                palindrome_better_banks=int((delta < 0).sum()), **intervals(delta),
                paired_sign_p=float(stats.binomtest(int((nonzero < 0).sum()), len(nonzero), .5).pvalue) if len(nonzero) else 1.))
    for row, corrected in zip(contrasts, holm([r['paired_sign_p'] for r in contrasts])):
        row['paired_sign_holm_ten'] = corrected
    out.mkdir(parents=True, exist_ok=False)
    figs = out/'figures'
    figs.mkdir()
    result = dict(passed=True, rows=list(rows.values()), contrasts=contrasts,
        units='eight shared sampling banks at fixed trained checkpoints; resolutions are repeated measurements',
        decisions=dict(task_general_palindrome_gain='reject', Ring8_pair_ordering='survives all registered settings',
                       Product16_pair_ordering='alternating remains lower at all registered settings'),
        native_verification_sha256=sha(verified/'COMPLETE.json'), reference_verification_sha256=sha(reference/'COMPLETE.json'),
        reference_scores_sha256=sha(reference/'scores.json'), source_sha256=sha(Path(__file__)))
    dump(out/'comparison.json', result)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    labels = {PAL: 'Palindrome', ALT: 'Alternating', SHARED: 'Shared P'}
    colors = {PAL: '#1b9e77', ALT: '#d95f02', SHARED: '#2166ac'}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7), constrained_layout=True)
    for i, task in enumerate(TASKS):
        for j in range(2):
            for family in [PAL, ALT, SHARED]:
                settings = [(k, 128) for k in [4, 8, 16]] if j == 0 else [(4, k) for k in [128, 256, 512]]
                present = [(k, rows[(task, family, *k)]['score']) for k in settings if (task, family, *k) in rows]
                axes[i, j].plot([k[j] for k, _ in present], [v for _, v in present], 'o-', label=labels[family], color=colors[family])
            axes[i, j].set(xscale='log', xticks=[4, 8, 16] if j == 0 else [128, 256, 512],
                xticklabels=['4', '8', '16'] if j == 0 else ['128', '256', '512'],
                xlabel='Internal steps (Heun128 fixed)' if j == 0 else 'Outer Heun steps (internal4 fixed)',
                ylabel='Pooled SW2', title='Ring8' if i == 0 else 'Product16')
            axes[i, j].legend(frameon=False)
            axes[i, j].xaxis.set_minor_locator(NullLocator())
    for ext in ['png', 'pdf']:
        fig.savefig(figs/('resolution-comparison.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for i, task in enumerate(TASKS):
        group = [r for r in contrasts if r['dataset'] == task]
        for k, row in enumerate(group):
            lo, hi = row['marginal95']
            axes[i].hlines(k, lo, hi, color='#2166ac', lw=2)
            axes[i].scatter(row['bank_differences'], k+np.linspace(-.06, .06, 8), color='.7', s=12)
            axes[i].scatter(row['paired_mean'], k, color='black', s=25)
        axes[i].axvline(0, color='.4', lw=.8)
        axes[i].set(yticks=range(5), yticklabels=[f"inner{r['inner']}/Heun{r['outer']}" for r in group],
            xlabel='Paired SW2 difference (palindrome minus alternating)', title='Ring8' if i == 0 else 'Product16')
    for ext in ['png', 'pdf']:
        fig.savefig(figs/('paired-resolution-effects.'+ext), dpi=180)
    plt.close(fig)
    table = '| Task | Internal steps | Heun steps | Palindrome | Alternating | Shared P |\n|---|---:|---:|---:|---:|---:|\n'
    for row in contrasts:
        key = (row['dataset'], SHARED, row['inner'], row['outer'])
        shared = f"{rows[key]['score']:.3f}" if key in rows else '---'
        table += f"| {'Ring8' if row['dataset']==TASKS[0] else 'Product16'} | {row['inner']} | {row['outer']} | {row['palindrome']:.3f} | {row['alternating']:.3f} | {shared} |\n"
    (out/'table.md').write_text(table)
    dump(out/'COMPLETE.json', dict(passed=True, unique_new_cells=20, repeated_anchor_rows=4,
        prior_shared_reference_cells=8, primary_resolution_contrasts=10,
        artifacts={str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()}))
    print('complete matched order numerical analysis:', out)


if __name__ == '__main__':
    main()
