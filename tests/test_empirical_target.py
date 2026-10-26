import unittest

import numpy as np

from scripts.diagnose_empirical_target import empirical_teacher


class EmpiricalTargetTests(unittest.TestCase):
    def test_single_atom_has_deterministic_conditional_velocity(self):
        atoms=np.array([[.7,-1.3]])
        x=np.array([[.2,.5],[-.1,.2]])
        result=empirical_teacher(atoms,x,.4)
        np.testing.assert_allclose(result['velocity'],(atoms-x)/.6,atol=1e-14)
        np.testing.assert_allclose(result['jacobian'],np.broadcast_to(-np.eye(2)/.6,(2,2,2)),atol=1e-13)
        np.testing.assert_allclose(result['noise_covariance'],0.,atol=1e-13)
        np.testing.assert_array_equal(result['ess'],np.ones(2))

    def test_posterior_jacobian_matches_independent_finite_difference(self):
        rng=np.random.default_rng(0)
        atoms=rng.normal(size=(13,3))
        x=rng.normal(size=(21,3))
        for time in (.1,.5,.9):
            result=empirical_teacher(atoms,x,time)
            columns=[]
            for axis in range(3):
                delta=np.zeros_like(x);delta[:,axis]=1e-5
                columns.append((empirical_teacher(atoms,x+delta,time)['velocity']-
                    empirical_teacher(atoms,x-delta,time)['velocity'])/(2e-5))
            numerical=np.stack(columns,axis=-1)
            np.testing.assert_allclose(numerical,result['jacobian'],atol=2e-5,rtol=2e-6)
            np.testing.assert_allclose(result['noise_covariance'],
                (np.eye(3)[None]+(1-time)*result['jacobian'])/time,atol=1e-12,rtol=1e-10)


if __name__=='__main__':
    unittest.main()
