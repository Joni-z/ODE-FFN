"""Complete-network rank limit and best-correction floor, independent of fitting."""
import unittest
import numpy as np
import torch
from synthetic.models import VelocityNet
from scripts.check_gate_geometry import gate_context,jacobian
from scripts.diagnose_clock_rank import rank_tail_floor


class ClockRankTests(unittest.TestCase):
    def test_svd_floor_is_attained_by_best_rank_four_matrix(self) -> None:
        residual=np.diag(np.arange(8,0,-1,dtype=float))[None]
        self.assertEqual(float(rank_tail_floor(residual,4)[0]),30.)
        correction=residual.copy();correction[:,4:,4:]=0
        np.testing.assert_allclose(np.sum((residual-correction)**2,axis=(1,2)),rank_tail_floor(residual,4))

    def test_complete_scalar_network_correction_respects_four_gate_bound(self) -> None:
        torch.set_num_threads(1);torch.manual_seed(0)
        model=VelocityNet(8,'ode_original',width=12,depth=2,gate_hidden=12,orders=1,tau=1.).double()
        with torch.no_grad():model.output.weight.normal_()
        x=torch.randn(16,8,dtype=torch.float64);t=torch.full((16,1),.5,dtype=torch.float64)
        values={}
        with gate_context(model,values,capture=True):p,full=jacobian(model,x,t)
        with gate_context(model,values):q,anchor=jacobian(model,x,t)
        torch.testing.assert_close(p,q,atol=0,rtol=0)
        singular=torch.linalg.svdvals(full-anchor)
        self.assertGreater(float(singular[:,0].max()),1e-5)
        self.assertLess(float(singular[:,4:].max()),1e-12)


if __name__=='__main__':unittest.main()
