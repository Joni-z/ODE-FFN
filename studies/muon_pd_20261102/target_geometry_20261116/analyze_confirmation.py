"""Source-coverage audit and figures for every independent confirmation pair."""
import argparse
import itertools
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, sha, dump, TIMES


def exact_bank_intervals(difference: np.ndarray) -> dict:
    """Enumerate sorted bootstrap indices with exact integer multiplicities."""
    values, weights = [], []
    for indices in itertools.combinations_with_replacement(range(8), 8):
        counts = np.bincount(indices, minlength=8)
        weights.append(math.factorial(8)//math.prod(math.factorial(int(k)) for k in counts))
        values.append(math.fsum(float(difference[i]) for i in indices)/8)
    assert len(values) == 6435 and sum(weights) == 8**8
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(weights, dtype=np.int64)[order])
    ordered = np.asarray(values)[order]
    at = lambda q: float(ordered[np.searchsorted(cdf, math.ceil(q*8**8))])
    return dict(bootstrap95=[at(.025), at(.975)], two_contrast_bonferroni=[at(.0125), at(.9875)],
                distinct_resample_multisets=6435, ordered_resamples=8**8)


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['input', 'study', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    root, study, out = map(Path, [args.input, args.study, args.output])
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    terminal = read(root/'terminal-receipts.json')
    assert terminal['passed'] and all(v == 0 for v in terminal['native_exits'].values())
    assert all(v == 0 for v in terminal['independent_exits'].values())
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    anchors = read(study/'anchors.json')
    assets = Path(anchors['assets'])
    manifest = read(assets/'manifest.json')
    assert sha(assets/'manifest.json') == anchors['assets_manifest_sha256']
    summaries, pairs, coverage = [], [], []
    for label, dataset in [('ring', 'gmm_ring8_2d'), ('product', 'rotated_product_bimodal_16d')]:
        native, verified = root/label, root/(label+'-verified')
        complete, proof = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json')
        assert complete['passed'] and proof['passed'] and not complete['witness'] and not proof['witness']
        assert complete['split'] == 'confirmation' and complete['banks'] == 8
        assert complete['cases'] == proof['cases'] == 2 and complete['rows'] == proof['rows'] == 10
        assert complete['points_per_time'] == 32768 and complete['times'] == list(TIMES)
        result = read(verified/'summary.json')
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert result['input_complete_sha256'] == sha(native/'COMPLETE.json')
        source = next(r for r in manifest['rows'] if r['dataset'] == dataset and r['split'] == 'confirmation')
        assert sha(Path(source['file'])) == source['sha256']
        with np.load(source['file']) as z:
            source_array = np.concatenate([z[f'source_{b}'] for b in range(8)]).astype(float)
            target_array = np.concatenate([z[f'target_{b}'] for b in range(8)]).astype(float)
        for case in read(native/'rows.json')['cases']:
            assert case['assets_sha256'] == source['sha256']
            file = native/case['raw_file']
            assert sha(file) == case['raw_sha256']
            with np.load(file) as z:
                assert np.array_equal(z['source'], source_array)
                assert np.array_equal(z['target'], target_array)
                assert np.array_equal(z['bank_index'], np.repeat(np.arange(8), 4096))
                assert np.array_equal(z['time'], np.asarray(TIMES))
                assert z['jacobian'].shape == (5, 32768, source_array.shape[1], source_array.shape[1])
            coverage.append(dict(id=case['id'], pairs=32768, times=5, banks=8,
                exact_all_source_target_arrays=True, raw_sha256=case['raw_sha256'], assets_sha256=source['sha256']))
        pair = result['matched_pairs'][0]
        assert pair['dataset'] == dataset and pair['resampling_unit'] == 'sampling_bank' and pair['resampling_units'] == 8
        difference = np.asarray(pair['bank_differences'])
        assert abs(difference.mean()-pair['primary_gap']) < 1e-11
        pair['exact_bootstrap'] = exact_bank_intervals(difference)
        pair['expected_direction_banks'] = int(np.sum(difference*pair['expected_gap_sign'] > 0))
        pairs.append(pair)
        summaries.extend(result['rows'])
        for src, name in [(native/'COMPLETE.json', label+'-native-COMPLETE.json'),
            (native/'rows.json', label+'-native-rows.json'), (verified/'COMPLETE.json', label+'-independent-COMPLETE.json'),
            (verified/'summary.json', label+'-summary.json')]:
            shutil.copyfile(src, out/name)
    all_points_match = all(p['predicted_direction_holds'] for p in pairs)
    resolved = all(p['expected_gap_sign']*p['exact_bootstrap']['two_contrast_bonferroni'][0] > 0 and
                   p['expected_gap_sign']*p['exact_bootstrap']['two_contrast_bonferroni'][1] > 0 for p in pairs)
    conclusion = ('The focal direction confirms on both tasks with all independent inputs.' if all_points_match and resolved else
                  'The full independent comparison does not resolve the focal association on both tasks.')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8.2, 4), constrained_layout=True)
    for i, pair in enumerate(pairs):
        differences = np.asarray(pair['bank_differences'])
        lo, hi = pair['exact_bootstrap']['two_contrast_bonferroni']
        ax.scatter(differences, i+np.linspace(-.08, .08, 8), color='.6', s=28, label='8 common banks' if i == 0 else None)
        ax.hlines(i, lo, hi, color='#2166ac', lw=3, label='Exact two-contrast interval' if i == 0 else None)
        ax.scatter([pair['primary_gap']], [i], color='black', zorder=3, s=35)
    ax.axvline(0, color='.4', lw=1)
    ax.set(yticks=[0, 1], yticklabels=['Ring8', 'Product16'],
           xlabel='Full Jacobian-error difference (palindrome minus alternating)', ylim=(-.4, 1.4))
    ax.legend(fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('independent-bank-differences.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, pair, title in zip(axes, pairs, ['Ring8', 'Product16']):
        rows = [r for r in summaries if r['dataset'] == pair['dataset']]
        pal = next(r for r in rows if r['family'].startswith('palindrome'))
        alt = next(r for r in rows if r['family'].startswith('alternating'))
        delta = [p['velocity_jacobian_error']-a['velocity_jacobian_error'] for p, a in zip(pal['by_time'], alt['by_time'])]
        ax.plot(TIMES, delta, 'o-', color='#2166ac')
        ax.axhline(0, color='.4', lw=1)
        ax.set(title=title, xlabel='Generative interpolation time t', ylabel='Jacobian-error difference (PAL - ALT)')
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('independent-time-differences.'+ext), dpi=180)
    plt.close(fig)
    table = '| Task | PAL J error | ALT J error | Difference | Exact adjusted interval | Banks | PAL SW2 | ALT SW2 |\n|---|---:|---:|---:|---|---:|---:|---:|\n'
    for p in pairs:
        task = 'Ring8' if p['dataset'] == 'gmm_ring8_2d' else 'Product16'
        lo, hi = p['exact_bootstrap']['two_contrast_bonferroni']
        pal = next(r for r in summaries if r['dataset'] == p['dataset'] and r['family'].startswith('palindrome'))
        alt = next(r for r in summaries if r['dataset'] == p['dataset'] and r['family'].startswith('alternating'))
        table += f"| {task} | {p['palindrome']:.3f} | {p['alternating']:.3f} | {p['primary_gap']:.3f} | [{lo:.3f}, {hi:.3f}] | {p['expected_direction_banks']}/8 | {pal['confirmation_pooled_sw2']:.3f} | {alt['confirmation_pooled_sw2']:.3f} |\n"
    (out/'table.md').write_text(table)
    (out/'analysis-report.md').write_text(f'''# Independent target-response confirmation

**{conclusion}** This tests a within-pair association, not a general ODE advantage.

The selection observation fixed the metric before this confirmation: average
||J_v-J_v*||_F²/d, equal point/time weights. We retain all8x4096source/target
pairs from each preexisting confirmation cohort at t=.1,.3,.5,.7,.9. Four
unchanged20kseed0checkpoints give655360evaluated interpolation states. These
are five views of32768pairs per task, not independent trajectories or trainings.

{table}

![All eight paired banks](figures/independent-bank-differences.png)

Dots are paired bank differences; bars are exact97.5% percentile bootstrap
intervals for the two-contrast family. Negative is the predicted Ring8direction;
positive is the predicted Product16direction. All bank values are retained in
summary.json. Registered10000-draw intervals and exhaustive sensitivity intervals
are both reported; exhaustive enumeration changes no population or metric.

![Every registered time](figures/independent-time-differences.png)

The second figure preserves all five time slices. It describes where the
focal response difference occurs and does not substitute a selected time for
the registered average. These states follow independent linear interpolation,
not the actual learned sampling trajectory.

## What this changes

The full-output derivative is a concrete bridge from a hidden composition
constraint to the learned task field. A confirmed association would establish
that the small selection pattern survives independent inputs. It still does
not show that correcting this derivative causes the sample improvement.
The selection cohort's Product16SwiGLU samples better despite greater average
Jacobian error, so the metric cannot rank all architectures or account for
an ODE-specific advantage. The next decisive measurement is the transported
effect of field error along common oracle paths, including its mean component.

## Verification

Both native observation processes and both independent verifiers exited0.
All source/target arrays equal the complete bound NPZbanks exactly, including
bank identity/order. Every raw derivative file is hashed; both task COMPLETE
receipts and verifier reconstructions are included. The input-dependent target
identities, wrong-object controls and sampling counterexamples remain in the
preceding derivation/readiness bundle. No training or image source was changed.

## Claim candidates

- Claim: the focal average-response ordering persists on independent inputs.
  Evidence: four full arrays, every paired bank, exact intervals. Decision:
  {'keep within this fixed-checkpoint comparison' if all_points_match and resolved else 'unresolved or rejected where direction/precision fails'}.
- Claim: output-Jacobian error alone explains generation quality across models.
  Evidence: ordinary Product16counterexample in the selection study. Decision:
  reject that general ranking; test trajectory-weighted error next.
- Claim: hidden ODE structure is uniquely responsible for a generation gain.
  Evidence: no matched causal output-response intervention here. Decision: not established.
''')
    (out/'stats-appendix.md').write_text('''# Conditional input uncertainty

Fixed checkpoints,32768paired source/target inputs per task, five repeated times.
The eight banks are the paired resampling units. Registered percentile bootstrap:
10000draws,RNG0,95% and97.5%two-contrast intervals. Raw primary outputs remain
in ring-summary.json/product-summary.json. An additional exact sensitivity check
enumerates6435sorted resample multisets with integer multiplicities summing to
8^8ordered resamples. Its intervals are reported alongside the original draws.
No training-repeat uncertainty or across-model significance is inferred. The
small number of banks limits tail resolution even when every source is retained.
No normality assumption, t-test or p-value is used. Stage-order weights and the
generation comparison are unchanged from the earlier registered experiment.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalogue

## independent-bank-differences.png / .pdf

Purpose: test both prespecified primary directions on independent complete banks.
Data: four verified checkpoint arrays, eight common banks per task. Dots are bank
differences; black points are all-source means; bars are exact conditional97.5%
bootstrap intervals. Notice sign and bank consistency. A resolved predicted sign
supports the narrow association; it does not establish training variability or cause.

## independent-time-differences.png / .pdf

Purpose: locate the full independent response differences across all five times.
Data: per-time verified summaries using all32768pairs. No error bars; times reuse
the same pairs. Notice which slices contribute without selecting a favorable one.
This informs a future common-trajectory test, not a new architecture claim.
''')
    dump(out/'summary.json', dict(passed=True, all_point_directions_match=all_points_match,
        both_exact_adjusted_intervals_resolved=resolved, pairs=pairs, rows=summaries, source_coverage=coverage))
    dump(out/'COMPLETE.json', dict(passed=True, cases=4, pairs_per_task=32768, times=5,
        evaluated_interpolation_states=655360, source_coverage_exact=True, analysis_source_sha256=sha(Path(__file__)),
        artifacts={str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()}))
    print(conclusion, flush=True)
    print(table, flush=True)


if __name__ == '__main__':
    main()
