"""Independent NumPy audit of one native-time-resolution case."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def comparison(a: np.ndarray, b: np.ndarray) -> dict:
    d = a-b
    return dict(rms=float(np.sqrt(np.einsum('nd,nd->', d, d)/d.size)),
        maximum=float(np.abs(d).max()),
        componentwise_pass=bool(np.all(np.abs(d) <= 1e-4+1e-3*np.abs(b))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, row = read(root/'COMPLETE.json'), read(root/'rows.json')
    assert complete['passed'] and sha(root/'rows.json') == complete['rows_sha256']
    count = 1 if complete['witness'] else 32
    assert complete['sources'] == 8*count and complete['banks'] == 8
    assert sha(Path(row['assets_file'])) == row['assets_sha256']
    with np.load(row['assets_file']) as z:
        source = np.concatenate([z[f'source_{b}'][:count] for b in range(8)]).astype(float)
    endpoints, maximum = {}, 0.
    actual = [int(v['steps']) for v in row['levels']]
    assert actual in [[128, 1024, 1000, 1032, 2048, 4096], [128, 1024, 1000, 1032, 2048, 4096, 8192]]
    for level in row['levels']:
        file = root/level['file']
        assert sha(file) == level['sha256']
        with np.load(file) as archive:
            value = {key: archive[key].copy() for key in archive.files}
        assert all(np.isfinite(v).all() for v in value.values())
        assert np.array_equal(value['source'], source)
        assert np.array_equal(value['bank_index'], np.repeat(np.arange(8), count))
        assert value['path'].shape == (9, 8*count, source.shape[1])
        assert np.array_equal(value['path'][0], source) and np.array_equal(value['path'][-1], value['endpoint'])
        if level['reused']:
            original = Path(level['binding']['file'])
            assert sha(original) == level['binding']['sha256'] and level['actual_new_field_calls'] == 0
            indices = np.concatenate([np.arange(count)+b*32 for b in range(8)])
            with np.load(original) as z:
                assert np.array_equal(z['endpoints'][-1][indices], value['endpoint'])
                assert np.array_equal(z['paths'][-1][:, indices], value['path'])
        else:
            assert level['actual_new_field_calls'] == 2*level['steps']
        endpoints[level['steps']] = value['endpoint']
        if level['steps'] in [4096, 8192]:
            measured = comparison(endpoints[level['steps']//2], value['endpoint'])
            for key, v in measured.items():
                assert np.isclose(v, level['qualification'][key], atol=1e-12, rtol=1e-11)
            assert not measured['componentwise_pass'] or level == row['levels'][-1]
    final = row['final_steps']
    assert final == complete['final_steps'] == max(endpoints)
    gate = comparison(endpoints[final//2], endpoints[final])
    assert gate['componentwise_pass'] == row['qualified'] == complete['qualified']
    assert row['qualified'] or final == 8192
    contrasts = []
    for contrast in row['contrasts']:
        left, right = contrast['left'], contrast['right']
        measured = comparison(endpoints[left], endpoints[right])
        for key, v in measured.items():
            maximum = max(maximum, abs(float(v)-float(contrast[key])))
            assert np.isclose(v, contrast[key], atol=1e-12, rtol=1e-11)
        d = endpoints[left]-endpoints[right]
        banks = [float(np.sqrt(np.square(d[b*count:(b+1)*count]).mean())) for b in range(8)]
        assert np.allclose(banks, contrast['bank_rms'], atol=1e-12, rtol=1e-11)
        contrasts.append(dict(left=left, right=right, **measured, bank_rms=banks,
            per_source_rms=np.sqrt(np.mean(d*d, axis=1)).tolist(), per_source_max=np.abs(d).max(axis=1).tolist()))
    result = dict(passed=True, witness=complete['witness'], id=row['id'], dataset=row['dataset'], family=row['family'],
        qualified=row['qualified'], final_steps=final, final_gate=gate, contrasts=contrasts,
        all_source_arrays_exact=True, max_reconstruction_error=maximum,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(passed=True, witness=complete['witness'], id=row['id'],
        summary_sha256=sha(out/'summary.json'), max_error=maximum), indent=2)+'\n')
    print(json.dumps(dict(passed=True, id=row['id'], qualified=row['qualified'], max_error=maximum)), flush=True)


if __name__ == '__main__':
    main()
