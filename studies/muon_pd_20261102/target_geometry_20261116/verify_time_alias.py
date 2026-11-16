"""Independent numerical reconstruction of the time-alias counterexample."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete = json.loads((root/'COMPLETE.json').read_text())
    assert complete['passed'] and complete['cases'] == 12
    assert sha(root/'rows.json') == complete['rows_sha256']
    rows = json.loads((root/'rows.json').read_text())['rows']
    maximum = 0.
    for row in rows:
        file = root/row['raw_file']
        assert sha(file) == row['raw_sha256']
        with np.load(file) as archive:
            a = {k: archive[k].copy() for k in archive.files}
        assert len(a['original_field']) == row['field_points']
        assert len(a['original_grid']) == 1025 and len(a['original_endpoint']) == 16
        assert np.array_equal(a['grid_t'], np.arange(1025)[:, None]/1024)
        assert row['original_frequency']-row['alias_frequency'] == 1024
        for n in [128, 256, 512, 1024]:
            t = np.arange(n+1)/n
            first = 2*np.pi*row['original_frequency']*t
            second = 2*np.pi*row['alias_frequency']*t
            assert np.max(np.abs(np.sin(first)-np.sin(second))) < 5e-12
            assert np.max(np.abs(np.cos(first)-np.cos(second))) < 5e-12
        assert np.allclose(a['original_grid'], a['alias_grid'], atol=1e-9, rtol=1e-9)
        assert np.allclose(a['original_endpoint'], a['alias_endpoint'], atol=1e-9, rtol=1e-9)
        assert np.allclose(a['original_grid32'], a['alias_grid32'], atol=1e-4, rtol=1e-4)
        assert np.allclose(a['original_endpoint32'], a['alias_endpoint32'], atol=3e-4, rtol=3e-4)
        dimension = a['original_field'].shape[1]
        difference = a['original_field']-a['alias_field']
        fitting = a['original_field']-a['oracle']
        alias_fitting = a['alias_field']-a['oracle']
        delta = float(np.einsum('nd,nd->', difference, difference)/len(difference)/dimension)
        mse = float(np.einsum('nd,nd->', fitting, fitting)/len(fitting)/dimension)
        alias_mse = float(np.einsum('nd,nd->', alias_fitting, alias_fitting)/len(fitting)/dimension)
        expected = dict(off_grid_difference_mse=delta, native_oracle_field_mse=mse,
            alias_oracle_field_mse=alias_mse, off_grid_change_over_fitting_error_rms=float(np.sqrt(delta/mse)),
            grid_max_difference=float(np.max(np.abs(a['original_grid']-a['alias_grid']))),
            endpoint_max_difference=float(np.max(np.abs(a['original_endpoint']-a['alias_endpoint']))),
            fp32_grid_max_difference=float(np.max(np.abs(a['original_grid32']-a['alias_grid32']))),
            fp32_endpoint_max_difference=float(np.max(np.abs(a['original_endpoint32']-a['alias_endpoint32']))))
        for prefix in ['original', 'alias']:
            difference = a[prefix+'_jacobians']-a['oracle_jacobians']
            expected['selection_'+prefix+'_jacobian_error'] = float(np.einsum('tnij,tnij->', difference, difference)/5/128/dimension)
        for key, value in expected.items():
            maximum = max(maximum, abs(value-row[key]))
            assert np.isclose(value, row[key], atol=1e-12, rtol=1e-11), (row['id'], key)
    proof = dict(passed=True, cases=12, max_error=maximum,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    (out/'COMPLETE.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
