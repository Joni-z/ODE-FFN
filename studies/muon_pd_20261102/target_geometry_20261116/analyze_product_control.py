"""Exact-oracle and actual-law null evidence, separate from learned-model results."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from geometry import read, dump, sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    quantile, gauge, verified = root/'product-quantile-v2', root/'product-gauge', root/'product-control-verified'
    q, g, v = [read(folder/'summary.json') for folder in [quantile, gauge, verified]]
    for folder in [quantile, gauge, verified]:
        proof = read(folder/'COMPLETE.json')
        assert proof['passed'] and proof['sources'] == 256
        assert sha(folder/'summary.json') == proof['summary_sha256']
    assert v['quantile_input_complete_sha256'] == sha(quantile/'COMPLETE.json')
    assert v['gauge_input_complete_sha256'] == sha(gauge/'COMPLETE.json')
    exits = read(root/'math-control-exits.json')
    assert exits['actual_exits'] == [0, 0, 0]
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    for folder, label in [(quantile, 'quantile'), (gauge, 'gauge'), (verified, 'independent')]:
        for file in ['summary.json', 'COMPLETE.json']:
            shutil.copyfile(folder/file, out/(label+'-'+file))
    shutil.copyfile(root/'math-control-exits.json', out/'actual-exits.json')
    assert sha(quantile/'exact-oracle.npz') == q['analytic_arrays_sha256']
    with np.load(quantile/'exact-oracle.npz') as z:
        times, gain = z['times'].copy(), z['transition_to_endpoint'].copy()
    native = {r['steps']: r for r in q['archived_native_checks']}
    native.update({r['steps']: r for r in q['refined_native_checks']})
    levels = sorted(native)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    for key, label, color in [('endpoint_max', 'Endpoint maximum', '#2166ac'),
                               ('path_max', 'Nine-time path maximum', '#d95f02')]:
        axes[0].plot(levels, [native[n][key] for n in levels], 'o-', color=color, label=label)
    axes[0].set(xscale='log', yscale='log', xlabel='Native Heun steps', ylabel='Absolute error against exact quantile flow',
                title='Independent oracle reference')
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(levels, [str(n) for n in levels])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=8)
    for quantile_value, label, color in [(.5, 'Median', '#2166ac'), (.99, '99th percentile', '#1b9e77'), (1., 'Maximum', '#d95f02')]:
        values = np.quantile(gain, quantile_value, axis=(1, 2))
        axes[1].plot(times, values, 'o-', color=color, label=label)
    axes[1].set(yscale='log', xlabel='Physical time t', ylabel='Oracle transition derivative to endpoint',
                title='Highly unequal sensitivity')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('exact-oracle-and-sensitivity.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    lambdas = [r['coefficient'] for r in v['rows']]
    for key, label, color in [('displacement_mse', 'Actual paired displacement', '#2166ac'),
                               ('remainder_mse', 'Linear-prediction residual', '#d95f02')]:
        axes[0].plot(lambdas, [r[key] for r in v['rows']], 'o-', color=color, label=label)
    axes[0].set(yscale='log', xlabel='Error multiplier lambda', ylabel='Mean squared component displacement',
                title='Exact marginal-preserving control')
    axes[0].legend(fontsize=8)
    axes[1].plot(lambdas, [r['explained_fraction'] for r in v['rows']], 'o-', color='#762a83')
    axes[1].axhline(0, color='.5', lw=1)
    axes[1].set(xlabel='Error multiplier lambda', ylabel='First-order explained fraction',
                title='Correct law, failed linear endpoint prediction')
    for ax in axes:
        ax.set_xticks(lambdas, ['.125', '.25', '.5', '1'])
        ax.grid(axis='y', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('correct-law-failed-linear-prediction.'+ext), dpi=180)
    plt.close(fig)
    table = '| Error multiplier | Paired displacement MSE | Linear residual MSE | Explained fraction | Population marginals |\n|---:|---:|---:|---:|---|\n'
    for r in v['rows']:
        table += f"| {r['coefficient']:.3f} | {r['displacement_mse']:.3f} | {r['remainder_mse']:.3f} | {r['explained_fraction']:.3f} | Exactly native at every time |\n"
    (out/'table.md').write_text(table)
    last = q['refined_native_checks'][-1]
    (out/'analysis-report.md').write_text(f'''# Product16: exact flow and a distribution-preserving counterexample

**The actual Product16 law admits exact flows with nonzero velocity error and
failed first-order endpoint prediction, while every population marginal stays
correct.** This separates reference-coupled trajectory fidelity from generation.
It is a mathematical control on the project dataset, not a learned architecture.

## Exact oracle reference

In the known latent basis, both the target and Gaussian source factorize.
Each coordinate follows the unique scalar oracle flow
u_t=F_t^(-1)(Phi(u_0)). Its transition derivative from time a to b is
p_a(u_a)/p_b(u_b). The full derivation and assumptions are in
[the derivation](../product-quantile-derivation.md).

The check retains all256 original Product16 selection sources, all16 latent
coordinates and nine recorded times. The six archived model observations
have exactly the same oracle path at each common resolution, as expected.
At{last['steps']} steps the saved native oracle endpoint differs from the exact
reference by RMS {last['endpoint_rms']:.2e}, maximum {last['endpoint_max']:.2e}.
Time and source derivatives of the inverse-CDF path are checked independently
by finite differences. Direct CDF conservation error is at most
{v['quantile_cdf_conservation_max']:.2e}.

![Exact reference and sensitivity](figures/exact-oracle-and-sensitivity.png)

The left panel is an absolute oracle error check, rather than agreement of
two numerical solvers. The right panel retains every coordinate/source and
shows strongly unequal transport sensitivity. The largest source-to-endpoint
derivative is{q['maximum_source_to_endpoint_derivative']:.3f}; the largest
recorded time-to-endpoint derivative is{q['maximum_time_to_endpoint_derivative']:.3f}.
These are target properties on fixed inputs. They do not by themselves explain
the learned-model approximation failures.

## Same actual law, different trajectories

Write Psi_t for the oracle quantile map and use paths
Psi_t(exp(lambda*t*Omega)u_0), where Omega=.2(C-C^T) and C is the fixed cyclic
coordinate shift. Gaussian rotations preserve the initial law; applying Psi_t
therefore yields exactly the native marginal at every t and lambda. The added
velocity is e_i=g_i(Omega z)_i with g_i=phi(z_i)/p_t(y_i). Its weighted
divergence div(e)+e dot grad(log p_t) vanishes identically.

Omega, all256 sources and the four multipliers were fixed before the control.
Skew symmetry, orthogonality, weighted divergence, native flow velocity and
endpoint derivatives pass their analytic and finite-difference checks.
The derivative at lambda0 is nonzero. Yet the finite-scale linear prediction
is worse than predicting no paired displacement at all four multipliers:

{table}

![Correct law despite failed prediction](figures/correct-law-failed-linear-prediction.png)

Population correctness follows from the exact pushforward, not an empirical
SW2score. The finite256-source values quantify the failure of the source-coupled
linear approximation. Independent reconstruction uses the stored raw endpoints,
an independently coded mixture CDF/density and compensated energy sums; the
largest metric difference is{v['metric_reconstruction_max']:.2e}.

## Research decision

Continue the registered twelve-model calibration, preserving every outcome.
Its result decides whether the endpoint approximation is useful for each model.
It cannot decide whether a model generates the correct distribution. A later
ODE-specific explanation needs a distribution-relevant intervention in the
matched trained architectures. Closeness to one oracle coupling is insufficient.

The earlier Product16 CDF control was described as only an endpoint-law map.
For this factorized law it is also the exact native oracle endpoint; this
addendum corrects that interpretation. Older arrays, scores and finalized
reports remain unchanged. No such coordinatewise equivalence is assumed for Ring8.

## Claim candidates

- Claim: the Product16 oracle path equals its coordinatewise quantile flow.
  - Evidence: derivation, all-source CDF/velocity/derivative checks and saved native paths.
  - Allowed wording: exact under the native factorized target and isotropic Gaussian source.
  - Forbidden stronger wording: every mixture oracle has a coordinatewise quantile flow.
  - Uncertainty: finite inverse-CDF evaluation and native integration errors are separately recorded.
  - Next check: use the exact reference to interpret, rather than replace, trained-model evidence.
  - Decision: keep.
- Claim: failed first-order endpoint prediction proves poor generation.
  - Evidence: actual-law counterexample has identical population marginals and negative fractions.
  - Allowed wording: this implication is false, including on Product16.
  - Forbidden stronger wording: the learned models have this special density-preserving error.
  - Uncertainty: exact counterexample is not an attribution of a trained model's behavior.
  - Next check: qualified distribution-relevant matched intervention.
  - Decision: reject the implication.
''')
    (out/'stats-appendix.md').write_text('''# Units, controls and verification

All256 original selection sources and16 coordinates are retained. These are
deterministic evaluations of fixed analytic maps, not independent model fits.
Coordinates, times and error multipliers are repeated within each source;
no confidence interval, hypothesis test or model ranking is inferred from them.
Sensitivity percentiles describe this finite population of coordinate/source
pairs. The exact marginal identity is algebraic and does not require a sample
significance test. Positive/negative endpoint fractions concern a fixed coupling.

The four multipliers and cyclic skew matrix were fixed in the derivation before
running the control. No source or direction was selected to obtain a failure.
The quantile control uses bisection on strictly positive mixture densities and
preserves all input probabilities without clipping. Source and time derivatives
are checked at decreasing finite-difference sizes; every error is in the bound
summaries. The independent verifier reconstructs CDF conservation, transition
ratios, tangent coefficients, exact endpoint probabilities and all four energy
ratios without importing the control implementation.

All three actual author/check/verifier commands exit0. Raw arrays remain outside
Git; the complete hashes and measured check values are in the three summaries.
No new generation scores, training, optimizer or FFN changes are involved.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalog

## exact-oracle-and-sensitivity.png / .pdf
- Purpose: independently validate the oracle and expose target sensitivity variation.
- Data: quantile-summary.json, hash-bound exact-oracle.npz outside Git, all256sources.
- Caption: left shows absolute native-path errors versus the exact quantile flow; right shows descriptive median/99th/max over repeated source-coordinate pairs.
- Observation: second-order endpoint convergence and a large sensitivity tail coexist.
- Interpretation: oracle integration is checked independently; a uniformly useful finite linearization cannot be assumed.
- Limit: target sensitivity is not a demonstrated cause of learned-model error.

## correct-law-failed-linear-prediction.png / .pdf
- Purpose: falsify the implication from linear endpoint prediction failure to incorrect generation.
- Data: independent-summary.json, exact actual-law marginal-preserving control.
- Caption: four registered multipliers,256sources; population marginals stay exact by the Gaussian rotation pushforward; these are paired endpoint energies, not SW2orFID.
- Observation: every tested finite-scale fraction is negative despite correct marginals.
- Interpretation: distribution relevance must be tested separately from one reference coupling.
- Limit: the control does not prove trained errors are density preserving and is not a candidate architecture.
''')
    dump(out/'COMPLETE.json', dict(passed=True, analytic_control=True, sources=256,
        source_sha256=sha(Path(__file__)), files={str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


if __name__ == '__main__':
    main()
