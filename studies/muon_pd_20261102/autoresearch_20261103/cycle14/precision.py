"""Post-outcome precision diagnostic on saved primary arrays; no model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', required=True)
    p.add_argument('--analysis', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    runtime, analysis, out = Path(a.runtime_root), Path(a.analysis), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    replay = json.loads((analysis/'replay.json').read_text())
    assert replay['passed'] and replay['hosts'] == ['lenovo']
    grid = json.loads(Path(__file__).with_name('grid.json').read_text())
    cases = [c for c in grid['cells'] if c['host'] == 'lenovo']
    paths = {}
    for case in cases:
        paths[case['id']] = runtime/'lenovo/output/professor_validation14_20261104'/case['dataset']/case['id']/'seed000'
    arrays, identities = {}, []
    records = list(csv.DictReader((analysis/'recomputed.csv').open()))
    for case in cases:
        folder = paths[case['id']]
        with np.load(folder/'banks.npz', allow_pickle=False) as z:
            banks = {k: z[k].copy() for k in z.files}
        binding = json.loads((folder/'binding.json').read_text())
        if identities:
            assert binding['bank_identity'] == identities[0]
        identities.append(binding['bank_identity'])
        selected = sorted([r for r in records if r['cell'] == case['id'] and r['checkpoint_step']=='20000' and r['heun_steps']=='128'],key=lambda r:int(r['bank']))
        assert [int(r['bank']) for r in selected] == list(range(8))
        generated = []
        for row in selected:
            path = folder/row['generated_file']
            assert sha(path) == row['generated_sha256']
            generated.append(np.load(path, allow_pickle=False))
        arrays[case['id']] = generated
    directions = banks['directions'].astype(float)
    rows = []
    for omitted in [-1]+list(range(8)):
        keep = [b for b in range(8) if b != omitted]
        target = np.concatenate([banks[f'target_{b}'] for b in keep]).astype(float)
        target2 = np.concatenate([banks[f'target2_{b}'] for b in keep]).astype(float)
        target_sorted = np.sort(target @ directions.T, axis=0)
        floor = float(np.sqrt(np.mean((target_sorted-np.sort(target2 @ directions.T, axis=0))**2)))
        for case in cases:
            generated = np.concatenate([arrays[case['id']][b] for b in keep]).astype(float)
            assert np.isfinite(generated).all()
            score = float(np.sqrt(np.mean((np.sort(generated @ directions.T, axis=0)-target_sorted)**2)))
            rows.append(dict(cell=case['id'], family=case['family'], optimizer=case['optimizer'],
                projection_lr=case['projection_lr'], omitted_bank=omitted, samples=len(target),
                checkpoint_step=20000, heun_steps=128, sliced_w2=score, target_target_sw2=floor))
        print(f'pooled omitted={omitted}: {len(rows)}/45',flush=True)
    with (out/'pooled.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt=dict(passed=True,kind='post_outcome_precision_diagnostic',source_replay_sha256=sha(analysis/'replay.json'),
        source_rows_sha256=sha(analysis/'recomputed.csv'),producer_sha256=sha(__file__),
        primary_endpoint_unchanged=True,new_model_calls=0,new_training_runs=0,rows=len(rows),
        pooled_samples=32768,omission_samples=28672,independent_training_replicates=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    candidate=next(c for c in cases if c['family']=='pd_isru_k1')
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index,case in enumerate(c for c in cases if c!=candidate):
        def values(cell,heun):
            return {int(r['bank']):float(r['numpy_sw2']) for r in records if r['cell']==cell and r['checkpoint_step']=='20000' and r['heun_steps']==str(heun)}
        c,r=values(candidate['id'],128),values(case['id'],128)
        label=f"{case['family']} {case['optimizer']} {case['projection_lr']:g}"
        axes[0].plot(range(8),[c[b]-r[b] for b in range(8)],'o-',label=label,color=colors[index])
        candidate_pool={int(r['omitted_bank']):r['sliced_w2'] for r in rows if r['cell']==candidate['id']}
        reference_pool={int(r['omitted_bank']):r['sliced_w2'] for r in rows if r['cell']==case['id']}
        axes[1].plot(range(8),[candidate_pool[b]-reference_pool[b] for b in range(8)],'o-',color=colors[index])
        axes[1].axhline(candidate_pool[-1]-reference_pool[-1],color=colors[index],linestyle=':',alpha=.65)
    axes[0].set(title='Original primary bank comparisons',xlabel='Evaluation bank (4,096 samples)',ylabel='SW2: ISRU-D minus reference')
    axes[1].set(title='Post-outcome pooled sensitivity',xlabel='Omitted bank (28,672 samples)',ylabel='SW2: ISRU-D minus reference')
    for ax in axes:
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(range(8));ax.spines[['right','top']].set_visible(False)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'paired-banks-and-pooling.png',dpi=180)
    plt.close(fig)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
