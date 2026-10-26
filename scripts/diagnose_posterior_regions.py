"""Locate errors in exposed product16 evaluations using exact posterior geometry.

This is descriptive analysis of old checkpoints, not a fresh confirmation bank.
Partitions are fixed by the analytic posterior log-odds and derivative sign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(study: Path, data_root: Path, output: Path) -> dict:
    heldout = json.loads((study / 'heldout/summary.json').read_text())
    assert heldout['state'] == 'COMPLETE'
    assert sha256(data_root / 'manifest.json') == heldout['manifest_sha256']
    manifest = json.loads((data_root / 'manifest.json').read_text())
    dataset = 'rotated_product_bimodal_16d'
    spec = manifest['datasets'][dataset]['distribution']
    rotation = np.asarray(spec['rotation'])
    locations, variance = np.asarray(spec['locations']), np.asarray(spec['variances'])
    rows, cases, files = [], [], {}
    inputs = {}
    for model in heldout['rows']:
        assert model['dataset'] == dataset
        label = model['label']
        for time in (.1, .3, .5, .7, .9):
            relative = f'{dataset}/{label}/field/common-{time}.npz'
            path = study / 'heldout' / relative
            assert sha256(path) == heldout['files_sha256'][relative]
            files[relative] = sha256(path)
            with np.load(path, allow_pickle=False) as bank:
                x = bank['x']
                if time in inputs:
                    np.testing.assert_array_equal(inputs[time], x)
                else:
                    inputs[time] = x.copy()
                z = x @ rotation
                c = (1-time)**2 + time**2 * variance
                q = time * locations * z / c
                sigmoid = np.tanh(q)
                diagonal = (time*variance-(1-time))/c + time*(1-time)*locations**2/c**2*(1-sigmoid**2)
                oracle = np.einsum('ij,nj,kj->nik', rotation, diagonal, rotation)
                np.testing.assert_allclose(oracle, bank['oracle_jacobian'], atol=1e-10, rtol=1e-10)
                predicted = np.einsum('ij,njk,kl->nil', rotation.T, bank['jacobian'], rotation)
                residual = predicted.copy()
                coordinate = np.arange(len(locations))
                residual[:, coordinate, coordinate] -= diagonal
                row_error = (residual**2).sum(axis=-1)
                diagonal_error = np.diagonal(residual, axis1=-2, axis2=-1)**2
                total = float(row_error.sum()/len(x))
                recorded = float(np.mean(np.sum((bank['jacobian']-bank['oracle_jacobian'])**2,axis=(1,2))))
                np.testing.assert_allclose(total, recorded, atol=1e-10, rtol=1e-10)
                partitions = [('derivative_sign', name, mask) for name, mask in [
                    ('expanding', diagonal > 0), ('contracting_or_zero', diagonal <= 0)]]
                limits = [0., .5, 1., 2., 4., np.inf]
                partitions += [('absolute_half_log_odds', f'[{lo},{hi})', (np.abs(q) >= lo) & (np.abs(q) < hi))
                               for lo, hi in zip(limits[:-1], limits[1:])]
                for family, region, mask in partitions:
                    count = int(mask.sum())
                    contribution = float(row_error[mask].sum()/len(x))
                    rows.append(dict(label=label,time=time,partition=family,region=region,
                        coordinate_fraction=count/mask.size,coordinate_count=count,
                        jacobian_error_contribution=contribution,jacobian_error_share=contribution/total,
                        mean_row_squared_error=float(row_error[mask].mean()) if count else None,
                        diagonal_error_contribution=float(diagonal_error[mask].sum()/len(x)),
                        off_diagonal_error_contribution=float((row_error-diagonal_error)[mask].sum()/len(x)),
                        total_jacobian_error=total))
                for family in ['derivative_sign','absolute_half_log_odds']:
                    selected = [r for r in rows if r['label']==label and r['time']==time and r['partition']==family]
                    np.testing.assert_allclose(sum(r['jacobian_error_contribution'] for r in selected),total,atol=1e-10)
                if time == .7:
                    ordered = np.argsort(np.abs(q).ravel(), kind='stable')
                    choices = [('largest_oracle_derivative', int(np.argmax(diagonal))),
                               ('median_posterior_certainty', int(ordered[len(ordered)//2]))]
                    pred_field, true_field = bank['prediction']@rotation, bank['oracle']@rotation
                    for choice, flat in choices:
                        n, d = np.unravel_index(flat, q.shape)
                        cases.append(dict(selection=choice,label=label,time=time,sample=int(n),coordinate=int(d),
                            location=float(locations[d]),latent_x=float(z[n,d]),half_log_odds=float(q[n,d]),
                            positive_sign_probability=float((1+sigmoid[n,d])/2),
                            oracle_derivative=float(diagonal[n,d]),predicted_derivative=float(predicted[n,d,d]),
                            oracle_velocity=float(true_field[n,d]),predicted_velocity=float(pred_field[n,d])))
    result=dict(state='COMPLETE',rows=rows,cases=cases,source_sha256=sha256(Path(__file__)),
        heldout_summary_sha256=sha256(study/'heldout/summary.json'),files_sha256=files,
        interpretation='Exposed test diagnostics of frozen20000-step models; no independent confirmation or causal attribution. Coordinate partitions are repeated measurements, not training replicates.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.study,args.data_root,args.output)
    print(json.dumps(dict(state=result['state'],partition_rows=len(result['rows']),cases=result['cases']),indent=2))
