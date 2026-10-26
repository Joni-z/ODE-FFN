"""Rotated density, field and oracle Jacobian must transform consistently."""
import unittest
import numpy as np
from synthetic.data import suite
from scripts.check_gate_geometry import oracle_geometry


class RotatedMixtureTests(unittest.TestCase):
    def test_covariance_velocity_and_jacobian(self) -> None:
        dist=suite()['rotated_gmm_ring8_16d']
        x=np.random.default_rng(0).normal(size=(5,16))
        r=dist.rotation
        mean,cov=dist.moments()
        m,c=dist.base.moments()
        np.testing.assert_allclose(mean,r@m,atol=1e-14)
        np.testing.assert_allclose(cov,r@c@r.T,atol=1e-14)
        for t in (.1,.5,.9):
            result=oracle_geometry(dist,x,t)
            columns=[]
            for axis in range(16):
                d=np.zeros_like(x);d[:,axis]=1e-5
                columns.append((dist.velocity(x+d,t)-dist.velocity(x-d,t))/2e-5)
            np.testing.assert_allclose(result['oracle_jacobian'],np.stack(columns,axis=-1),atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(result['oracle_jacobian'],result['oracle_jacobian'].transpose(0,2,1),atol=1e-12)
            np.testing.assert_allclose(dist.log_density(x,t),dist.base.log_density(x@r,t),atol=1e-12)


if __name__=='__main__':unittest.main()
