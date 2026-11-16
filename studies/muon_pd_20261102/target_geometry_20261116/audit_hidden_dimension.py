"""Exact dimension reference on all retained hidden-state differences."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--energy', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, published, output = map(Path, [args.energy, args.summary, args.output])
    terminal = read(root/'terminal-receipts.json')
    assert terminal['passed'] and all(value == 0 for value in terminal['native_exits'].values())
    assert all(value == 0 for value in terminal['independent_exits'].values())
    frame = np.ones((1, 1))
    while len(frame) < 64:
        frame = np.block([[frame, frame], [frame, -frame]])
    frame /= 8
    assert np.array_equal(frame.T@frame, np.eye(64))
    rows, max_error, states = [], 0., 0
    for task in ['ring', 'product']:
        native = root/'formal'/task
        complete = read(native/'COMPLETE.json')
        assert complete['passed'] and complete['all_qualified'] and not complete['witness']
        assert sha(native/'rows.json') == complete['rows_sha256']
        for row in read(native/'rows.json')['rows']:
            total, normal = 0., 0.
            for layer in row['layers']:
                file = native/layer['file']
                assert sha(file) == layer['sha256']
                with np.load(file) as bundle:
                    delta = bundle['native_forward']-bundle['references'][-1]
                    reference = bundle['references'][-1]
                q = reference/np.sqrt(1+1.2*reference**2)
                norm2, dot = np.sum(q*q, axis=1), np.sum(delta*q, axis=1)
                normal += np.divide(dot*dot, norm2, out=np.zeros_like(norm2), where=norm2 != 0).sum()
                total += np.sum(delta*delta)
                observed, expected = np.mean((delta@frame)**2, axis=1), np.sum(delta*delta, axis=1)/64
                max_error = max(max_error, float(np.max(np.abs(observed-expected))))
                assert np.allclose(observed, expected, atol=1e-14, rtol=1e-14)
                states += len(delta)
            share = float(normal/total)
            old = next(item for item in read(published)['rows'] if item['id'] == row['id'])
            old_share = next(item for item in old['aggregate'] if item['steps'] == 4)['normal_squared_error_share']
            assert abs(share-old_share) < 1e-12
            rows.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'],
                normal_share=share, isotropic_reference=1/64, normal_enrichment=64*share))
    assert states == 2880 and len(rows) == 6
    dump(output, dict(passed=True, dimension=64, states=states, geometric_reference=1/64,
        max_exact_frame_identity_error=max_error, rows=rows, source_summary_sha256=sha(published),
        source_sha256=sha(Path(__file__)),
        scope='exact projection geometry; no random downstream intervention has been run'))
    print('PASS exact dimension reference:', states, 'states; frame error', max_error, flush=True)


if __name__ == '__main__':
    main()
