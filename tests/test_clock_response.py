"""Gate-response orientation is a separate constraint from correction rank."""
import unittest
import numpy as np
import torch
from synthetic.models import VelocityNet
from scripts.check_gate_geometry import gate_context,jacobian
from scripts.diagnose_clock_response import clock_response,project_response


class ClockResponseTests(unittest.TestCase):
    def test_wrong_response_plane_has_stronger_floor_than_rank_count(self) -> None:
        residual=np.diag([4.,3.,2.,1.])[None]
        response=np.eye(4)[:,2:][None]
        projected,rank=project_response(response,residual)
        self.assertEqual(int(rank[0]),2)
        self.assertEqual(float(np.sum((residual-projected)**2)),25.)
        self.assertEqual(float(np.sum(np.linalg.svd(residual,compute_uv=False)[:,2:]**2)),5.)

    def test_complete_clock_correction_lies_in_measured_response_span(self) -> None:
        torch.set_num_threads(1);torch.manual_seed(0)
        model=VelocityNet(8,'ode_original',width=12,depth=2,gate_hidden=12,orders=1,tau=1.).double()
        with torch.no_grad():model.output.weight.normal_()
        x=torch.randn(16,8,dtype=torch.float64);t=torch.full((16,1),.5,dtype=torch.float64)
        values={}
        with gate_context(model,values,capture=True):_,full=jacobian(model,x,t)
        with gate_context(model,values):_,anchor=jacobian(model,x,t)
        correction=(full-anchor).numpy()
        response=clock_response(model,x,t)
        projected,_=project_response(response,correction)
        np.testing.assert_allclose(projected,correction,atol=1e-12,rtol=1e-10)


if __name__=='__main__':unittest.main()
