"""Field observations and numerical gates; no eight-source population KL ranking."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from verify_density import read, sha, npz


FAMILIES = ['state_midpoint4', 'stage_operator_midpoint4', 'free_isru', 'swiglu64', 'swiglu192']
LABELS = {'state_midpoint4': 'Shared P', 'stage_operator_midpoint4': 'Stage P',
          'free_isru': 'Free ISRU', 'swiglu64': 'SwiGLU64', 'swiglu192': 'SwiGLU192'}
COLORS = ['#245c8c', '#c56035', '#45896a', '#8a659e']
TASKS = ['gmm_ring8_2d', 'rotated_product_bimodal_16d']


def main():
    p = argparse.ArgumentParser()
    for name in ['data', 'pilot', 'verified', 'refined', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    data_root, pilot, verified, refined, out = map(Path, [a.data, a.pilot, a.verified, a.refined, a.output])
    assert read(verified/'COMPLETE.json')['passed'] and read(refined/'COMPLETE.json')['passed']
    cases = read(verified/'verified-cases.json')
    final = read(refined/'qualified-cells.json')
    assert len(cases) == 8 and len(final) == 16
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 180, 'axes.spines.top': False, 'axes.spines.right': False})
    rows, ordered = [], []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        selected = [c for f in FAMILIES for c in cases if c['id'].startswith(task+'__'+f+'__')]
        assert len(selected) == 4
        for i, case in enumerate(selected):
            family = case['id'].split('__')[1]
            coords = []
            meta = read(data_root/'density-data.json')['datasets'][task]
            assert sha(data_root/meta['file']) == meta['sha256']
            data = npz(data_root/meta['file'])
            done = read(pilot/'cases'/case['id']/'COMPLETE.json')
            for mode, marker in [('original', 'o'), ('projected', 's')]:
                summary = next(f for f in case['field_summaries'] if f['mode'] == mode)
                record = next(f for f in done['fields'] if f['mode'] == mode)
                assert sha(pilot/'cases'/case['id']/record['file']) == record['sha256']
                values = npz(pilot/'cases'/case['id']/record['file'])
                bank_mse, bank_source = [], []
                for bank in range(4):
                    mask = data['field_banks'] == bank
                    bank_mse.append(float(values['field_mse'][mask].mean()))
                    bank_source.append(float(np.sqrt(np.mean(values['density_source'][mask]**2))))
                x, y = 1000*summary['field_mse'], summary['source_rms']
                coords.append((x, y))
                axs[task_index].scatter(1000*np.array(bank_mse), bank_source, s=13, marker=marker, color=COLORS[i], alpha=.22)
                axs[task_index].scatter(x, y, s=48, marker=marker, edgecolor=COLORS[i],
                                       facecolor=COLORS[i] if mode == 'original' else 'none', label=LABELS[family] if mode == 'original' else None)
                rows.append(dict(task=task, family=family, mode=mode, field_mse=summary['field_mse'],
                                 source_rms=y, field_mse_bank_mean=float(np.mean(bank_mse)),
                                 field_mse_bank_sd=float(np.std(bank_mse, ddof=1)), source_rms_bank_mean=float(np.mean(bank_source)),
                                 source_rms_bank_sd=float(np.std(bank_source, ddof=1))))
                initial = next(q for q in case['qualifications'] if q['mode'] == mode)
                final_row = next(q for q in final if q['id'] == case['id'] and q['mode'] == mode)
                ordered.append((('R8' if task_index == 0 else 'P16')+' '+LABELS[family]+' / '+mode, initial, final_row))
            axs[task_index].plot(*zip(*coords), color=COLORS[i], lw=.8, alpha=.6)
        axs[task_index].set_title('Ring8' if task_index == 0 else 'Product16')
        axs[task_index].set_xlabel('Field MSE x1,000 on fixed-time banks')
        axs[task_index].set_ylabel('Instantaneous density-source RMS')
        axs[task_index].legend(fontsize=8)
    fig.savefig(out/'figures/field-source.png')
    fig.savefig(out/'figures/field-source.pdf')
    plt.close(fig)
    columns = ['chain', 'logdet', 'ratio_refinement', 'endpoint_refinement']
    limits = np.array([.001, .001, .002, .001])
    matrices = [np.array([[row[index]['metrics'][key] for key in columns] for row in ordered])/limits for index in [1, 2]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
    for ax, values, title in zip(axs, matrices, ['Initial 512-step pilot', 'After targeted refinement']):
        im = ax.imshow(np.log10(np.maximum(values, 1e-4)), cmap='RdYlGn_r', norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=1), aspect='auto')
        for i in range(16):
            for j in range(4):
                ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', fontsize=7)
        labels = [name+(f" [{row['final_steps']}]" if title.startswith('After') else '') for name, initial, row in ordered]
        ax.set_yticks(range(16), labels, fontsize=7)
        ax.set_xticks(range(4), ['Chain', 'Local logdet', 'Log-ratio change', 'Endpoint change'], rotation=28, ha='right', fontsize=8)
        ax.set_title(title)
    fig.colorbar(im, ax=axs, shrink=.7, label='log10(error / tolerance); colors <=0 pass')
    fig.savefig(out/'figures/numerical-gates.png')
    fig.savefig(out/'figures/numerical-gates.pdf')
    plt.close(fig)
    with (out/'field-scores.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(passed=True, rows=rows, field_times=[.05, .25, .5, .75, .95], field_banks=4,
                   points_per_time_bank=64, trajectory_pilot_points=8, population_kl_ranking=False,
                   pilot_verified_sha256=sha(verified/'COMPLETE.json'), refined_verified_sha256=sha(refined/'COMPLETE.json'))
    (out/'comparison.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(passed=True, field_rows=len(rows), figures=2, output=str(out))))


if __name__ == '__main__':
    main()
