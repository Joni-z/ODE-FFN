"""Bound previous native endpoints and a batch-separable four-blend executor."""
from pathlib import Path
import numpy as np
from geometry import read, sha
from trajectory import LAMBDAS, integrate, torch

NEW_LEVELS = (1000, 1032, 2048, 4096, 8192)


def pilot_folder(pilot: Path, dataset: str) -> Path:
    return pilot/('ring' if dataset == 'gmm_ring8_2d' else 'product')


def cached_level(pilot: Path, row: dict, identity: dict, arrays: dict, steps: int, per_bank: int) -> tuple[dict, dict]:
    root = pilot_folder(pilot, row['dataset'])
    complete = read(root/'COMPLETE.json')
    checked = read(root.with_name(root.name+'-verified')/'COMPLETE.json')
    verification = root.with_name(root.name+'-verified')
    assert complete['passed'] and checked['passed'] and not complete['witness'] and not checked['witness']
    assert complete['cases'] == checked['cases'] == 6 and complete['sources_per_case'] == 256
    assert sha(verification/'summary.json') == checked['summary_sha256']
    assert read(verification/'summary.json')['input_complete_sha256'] == sha(root/'COMPLETE.json')
    assert sha(root/'rows.json') == complete['rows_sha256']
    meta = read(root/'rows.json')
    previous = next(r for r in meta['rows'] if r['id'] == row['id'])
    binding = next(r for r in meta['identities'] if r['id'] == row['id'])
    assert binding['checkpoint_sha256'] == identity['checkpoint_sha256']
    level = next(v for v in previous['levels'] if v['steps'] == steps)
    file = root/level['file']
    assert sha(file) == level['sha256']
    indices = np.concatenate([np.arange(per_bank)+bank*32 for bank in range(8)])
    expected = np.concatenate([arrays[f'source_{b}'][:32] for b in range(8)]).astype(float)
    with np.load(file) as z:
        assert np.array_equal(z['source'], expected)
        assert np.array_equal(z['lambdas'], LAMBDAS)
        result = dict(source=z['source'][indices], endpoint=z['endpoints'][-1][indices],
                      path=z['paths'][-1][:, indices], steps=np.asarray(steps))
    return result, dict(file=str(file), sha256=level['sha256'],
        original_rows_sha256=sha(root/'rows.json'), independent_completion_sha256=sha(root.with_name(root.name+'-verified')/'COMPLETE.json'))


def change(first: np.ndarray, second: np.ndarray) -> dict:
    residual = first-second
    return dict(rms=float(np.sqrt(np.mean(residual**2))), maximum=float(np.max(np.abs(residual))),
        componentwise_pass=bool(np.allclose(first, second, atol=1e-4, rtol=1e-3)))


class PackedBlends(torch.nn.Module):
    """Same four fields and sources, evaluated as one sample-separable batch."""
    def __init__(self, model, oracle, source_count: int):
        super().__init__()
        self.model, self.oracle = model, oracle
        coefficients = torch.tensor(LAMBDAS, dtype=torch.double).repeat_interleave(source_count)[:, None]
        self.register_buffer('coefficients', coefficients)

    def forward(self, x, t):
        assert len(x) == len(self.coefficients)
        value, true = self.model(x, t), self.oracle(x, t)
        mixed = true+self.coefficients*(value-true)
        return torch.where(self.coefficients == 1, value, mixed)


def packed_blends(model, oracle, source, steps: int) -> tuple[np.ndarray, np.ndarray]:
    endpoint, path = integrate(PackedBlends(model, oracle, len(source)), source.repeat(4, 1), steps)
    dim, n = source.shape[1], len(source)
    return endpoint.numpy().reshape(4, n, dim), path.numpy().reshape(9, 4, n, dim).transpose(1, 0, 2, 3)
