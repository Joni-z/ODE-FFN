"""Display the completed Product16 subcomparison while Ring8 finishes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_density import read, sha


def main():
    p = argparse.ArgumentParser()
    for key in ['statistics', 'geometry', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    stats, geom = read(a.statistics), read(a.geometry)
    assert geom['passed'] and len(geom['rows']) == 8 and stats['sources_per_model_field'] == 1024
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    families = ['state_midpoint4', 'stage_operator_midpoint4', 'free_isru', 'swiglu192']
    names = ['Shared P', 'Stage P', 'Free ISRU', 'SwiGLU']
    colors = ['#2166ac', '#d6604d', '#757575', '#222222']
    s = [next(v for v in stats['summaries'] if v['family'] == f and v['mode'] == 'projected') for f in families]
    g = [next(v for v in geom['rows'] if v['family'] == f and v['mode'] == 'projected') for f in families]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    x = np.arange(4)
    axes[0].bar(x, [v['mean_log_ratio'] for v in s], color=colors, width=.65,
                yerr=[v['mc_se'] for v in s], capsize=3)
    axes[0].set(ylabel='Estimated joint KL (nats)', title='Density error; error bars: one MC SE')
    axes[1].scatter(x, [v['normalized_conditional_variance'] for v in g], color=colors, s=60, zorder=3)
    axes[1].axhline(1, color='.4', linestyle='--', lw=1, label='Target = 1')
    axes[1].set(ylabel='Conditional variance / target variance', title='Broader components in shared P', ylim=(.9, 1.26))
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].bar(x, [v['posterior_sign_pair_covariance_rms'] for v in g], color=colors, width=.65)
    floor = geom['native_target_calibration']['posterior_sign_pair_covariance_rms']
    axes[2].axhline(floor, color='.4', linestyle='--', lw=1, label='Native target sample')
    axes[2].set(ylabel='Pairwise posterior-sign covariance RMS', title='No clear pairwise-dependence separation')
    axes[2].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x, names, rotation=15)
    fig.suptitle('Product16 after the same reflection projection; 1024 common source points')
    for suffix in ['png', 'pdf']:
        path = out/('product16-density-width.'+suffix)
        assert not path.exists()
        fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    (out/'product16-density-width-bindings.json').write_text(json.dumps(dict(
        statistics_sha256=sha(a.statistics), geometry_sha256=sha(a.geometry),
        plot_source_sha256=sha(__file__), sources_per_model=1024, mode='projected'), indent=2)+'\n')
    print(json.dumps(dict(passed=True, figure='product16-density-width', panels=3)))


if __name__ == '__main__':
    main()
