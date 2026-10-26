"""Exact factorized oracle against enumerated mixtures and finite differences."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from synthetic.data import RotatedProductMixture,RotatedGaussianMixture,suite
from synthetic.prepare import prepare
from synthetic.validate import validate
from synthetic.train import load_distribution
from scripts.check_gate_geometry import oracle_geometry


class ProductMixtureTests(unittest.TestCase):
    def test_matches_enumerated_gmm_including_endpoints_and_jacobian(self) -> None:
        rng=np.random.default_rng(0)
        rotation=np.linalg.qr(rng.standard_normal((3,3)))[0]
        locations=np.array([.7,1.3,2.])
        variances=np.array([.04,.1,.3])
        dist=RotatedProductMixture(locations,variances,rotation)
        means=np.array(list(itertools.product([-1,1],repeat=3)))*locations
        reference=RotatedGaussianMixture(means,np.tile(variances,(8,1)),np.ones(8),rotation)
        x=rng.standard_normal((31,3))
        for time in [0.,.13,.5,.87,1.]:
            np.testing.assert_allclose(dist.velocity(x,time),reference.velocity(x,time),atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(dist.log_density(x,time),reference.log_density(x,time),atol=2e-12,rtol=2e-12)
            actual=oracle_geometry(dist,x,time);expected=oracle_geometry(reference,x,time)
            np.testing.assert_allclose(actual['oracle_jacobian'],expected['oracle_jacobian'],atol=2e-11,rtol=2e-11)
            step=1e-5;columns=[]
            for coordinate in range(3):
                delta=np.eye(3)[coordinate]*step
                columns.append((dist.velocity(x+delta,time)-dist.velocity(x-delta,time))/(2*step))
            np.testing.assert_allclose(dist.jacobian(x,time),np.stack(columns,axis=-1),atol=2e-8,rtol=2e-8)

    def test_full_rank_input_dependent_part_and_population_moments(self) -> None:
        dist=suite()['rotated_product_bimodal_16d']
        x=np.ones((1,16))*.1@dist.rotation.T
        changed=dist.jacobian(x,.5)[0]-dist.jacobian(np.zeros_like(x),.5)[0]
        self.assertEqual(np.linalg.matrix_rank(changed,tol=1e-10),16)
        points=dist.sample(np.random.default_rng(0),100000)
        mean,cov=dist.moments()
        self.assertLess(np.max(np.abs(points.mean(0)-mean)/np.sqrt(np.diag(cov)/len(points))),7.)
        self.assertLess(np.linalg.norm(np.cov(points,rowvar=False,ddof=0)-cov)/np.linalg.norm(cov),.025)
        restored=RotatedProductMixture(**dist.to_dict())
        np.testing.assert_array_equal(restored.velocity(x,.5),dist.velocity(x,.5))

    def test_prepared_manifest_and_oracles_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=json.loads(Path('configs/synthetic/architecture_search/data_product16.json').read_text())
            config.update(train_size=64,validation_size=64,test_size=64,probe_size=64)
            p=root/'config.json';p.write_text(json.dumps(config));prepare(p,root/'data')
            manifest=validate(root/'data')
            spec=manifest['datasets']['rotated_product_bimodal_16d']
            self.assertEqual(spec['family'],'rotated_product_mixture')
            self.assertIsInstance(load_distribution(spec),RotatedProductMixture)


if __name__=='__main__':unittest.main()
