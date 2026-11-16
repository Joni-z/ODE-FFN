"""Bind complete data-independent frames before new downstream observations."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--readout', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, out = Path(args.readout), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    terminal = read(source/'terminal-receipts.json')
    assert terminal['passed'] and all(v == 0 for v in terminal['native_exits'].values())
    assert all(v == 0 for v in terminal['independent_exits'].values())
    rows = []
    for task in ['ring', 'product']:
        root = source/'formal'/task
        complete = read(root/'COMPLETE.json')
        assert complete['passed'] and not complete['witness'] and not complete['check_only']
        assert sha(root/'rows.json') == complete['rows_sha256']
        rows += [(root, row) for row in read(root/'rows.json')['rows']]
    rng = np.random.default_rng(0)
    manifest = []
    for root, row in sorted(rows, key=lambda pair: pair[1]['id']):
        frames = []
        for _ in row['layers']:
            q, r = np.linalg.qr(rng.standard_normal((64, 64)))
            signs = np.where(np.diag(r) >= 0, 1., -1.)
            frame = (q*signs).T
            assert np.allclose(frame@frame.T, np.eye(64), atol=1e-12, rtol=0)
            frames.append(frame)
        file = out/(row['id']+'.npz')
        np.savez_compressed(file, frames=np.stack(frames))
        manifest.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'],
            file=str(file), sha256=sha(file), layers=[item['layer'] for item in row['layers']],
            source_readout=str(root), source_rows_sha256=sha(root/'rows.json'),
            source_complete_sha256=sha(root/'COMPLETE.json')))
    assert len(manifest) == 6 and sum(len(row['layers']) for row in manifest) == 36
    dump(out/'manifest.json', dict(passed=True, evaluation_rng=0, dimension=64, cases=6,
        frames=36, directions_per_frame=64, rows=manifest,
        source_terminal_sha256=sha(source/'terminal-receipts.json'), source_sha256=sha(Path(__file__))))
    print('PASS bound frames:', len(manifest), 'cases, 36 frames,64 directions each', flush=True)


if __name__ == '__main__':
    main()
