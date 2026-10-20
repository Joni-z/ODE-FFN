"""Mathematical and reproducibility checks independent of torch/image dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from synthetic.data import GaussianMixture, suite, fm_batch
from synthetic.geometry import available_suite
from synthetic.prepare import prepare


class SyntheticTests(unittest.TestCase):
    def test_endpoints(self):
        rng = np.random.default_rng(5)
        for dist in suite().values():
            x = rng.normal(size=(17, dist.dim))
            mean, _ = dist.moments()
            np.testing.assert_allclose(dist.velocity(x, 0), mean - x, atol=1e-12)
            np.testing.assert_allclose(dist.velocity(x, 1), x, atol=1e-12)

    def test_continuity_equation(self):
        # Independent density/velocity consistency: d_t p + div(p*v) = 0.
        rng = np.random.default_rng(7)
        h = 1e-5
        for name, dist in suite().items():
            if dist.dim != 2:
                continue
            x = rng.normal(size=(32, 2)) * 2
            for t in (0.15, 0.5, 0.85):
                derivative = (np.exp(dist.log_density(x, t+h)) - np.exp(dist.log_density(x, t-h))) / (2*h)
                divergence = np.zeros(len(x))
                for axis in range(2):
                    offset = np.zeros_like(x)
                    offset[:, axis] = h
                    def flux(points):
                        return np.exp(dist.log_density(points, t)) * dist.velocity(points, t)[:, axis]
                    divergence += (flux(x+offset) - flux(x-offset)) / (2*h)
                np.testing.assert_allclose(derivative, -divergence, atol=2e-7, rtol=2e-5, err_msg=name)

    def test_stationary_gaussian(self):
        dist = GaussianMixture([[0,0]], [[1,1]], [1])
        # Independent interpolation has variance 0.5 at midpoint (not 1).
        x = np.array([[1.0, -2.0]])
        np.testing.assert_allclose(dist.velocity(x, 0.5), 0, atol=1e-14)
        np.testing.assert_allclose(dist.velocity(x, 0.25), -0.8*x, atol=1e-14)

    def test_sampling_moments(self):
        rng = np.random.default_rng(1)
        for dist in available_suite().values():
            x = dist.sample(rng, 100000)
            mean, cov = dist.moments()
            self.assertLess(np.max(np.abs(x.mean(0)-mean)/np.sqrt(np.diag(cov))), 0.02)
            self.assertLess(np.max(np.abs(x.var(0)/np.diag(cov)-1)), 0.03)

    def test_checkerboard_support(self):
        x = available_suite()['checkerboard_2d'].sample(np.random.default_rng(4), 10000)
        self.assertTrue(np.all((x >= -2) & (x < 2)))
        self.assertTrue(np.all(np.floor(x[:,0]).astype(int)%2 == np.floor(x[:,1]).astype(int)%2))

    def test_fm_pair_identity_and_resampling(self):
        x1 = suite()['gmm_ring8_2d'].sample(np.random.default_rng(1), 100)
        rng = np.random.default_rng(2)
        xt, t, v = fm_batch(x1, rng)
        np.testing.assert_allclose(xt+(1-t)*v, x1, atol=1e-6)
        self.assertFalse(np.array_equal(fm_batch(x1, rng)[0], xt))

    def test_reproducibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            config = json.loads(Path('configs/synthetic/data_v1.json').read_text())
            for key in ('train_size','validation_size','test_size','probe_size'):
                config[key] = 64
            p = root/'config.json'
            p.write_text(json.dumps(config))
            prepare(p, root/'first')
            prepare(p, root/'second')
            for name in config['datasets']:
                for split in ('train','validation','test','probe'):
                    with np.load(root/'first'/name/f'{split}.npz') as a, np.load(root/'second'/name/f'{split}.npz') as b:
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])
                with np.load(root/'first'/name/'train.npz') as a, np.load(root/'first'/name/'test.npz') as b:
                    self.assertFalse(np.array_equal(a['x1'],b['x1']))
            with self.assertRaises(FileExistsError):
                prepare(p, root/'first')


if __name__ == '__main__':
    unittest.main()
