"""Product-well checks catch collapse even when all samples have high density."""
import unittest
import numpy as np
import torch
from synthetic.data import RotatedProductMixture
from scripts.evaluate_product_holdout import sample_metrics


class ProductMetricTests(unittest.TestCase):
    def test_collapsed_well_is_detected_by_signs_and_marginal_distance(self) -> None:
        rng=np.random.default_rng(0)
        dist=RotatedProductMixture(np.ones(4),np.full(4,.04),np.eye(4))
        target=torch.from_numpy(dist.sample(rng,4096))
        independent=torch.from_numpy(dist.sample(rng,4096))
        collapsed=torch.ones_like(target)
        directions=torch.eye(4)
        good=sample_metrics(independent,target,directions,dist)
        bad=sample_metrics(collapsed,target,directions,dist)
        self.assertLess(good['mean_sign_imbalance'],.03)
        self.assertEqual(bad['mean_sign_imbalance'],.5)
        self.assertLess(good['sign_second_moment_off_diagonal_rms'],.05)
        self.assertEqual(bad['sign_second_moment_off_diagonal_rms'],1.)
        self.assertEqual(bad['off_well_coordinate_fraction'],0.)
        self.assertGreater(bad['mean_marginal_w2'],good['mean_marginal_w2'])
        # A high-density collapsed population has lower NLL, so NLL is not a win criterion.
        self.assertLess(bad['mean_target_negative_log_density'],good['mean_target_negative_log_density'])


if __name__=='__main__':unittest.main()
