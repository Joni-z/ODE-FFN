import unittest
import torch

from synthetic.interventions import action, evaluation_order
from synthetic.models import VelocityNet
from scripts.controlled_operators import controlled_family


class InterventionTests(unittest.TestCase):
    def test_nilpotent_exact_solution(self):
        # P^2=0 implies exp(sP)h=h+sPh for every s, including negative s.
        p=torch.tensor([[0.,3.],[0.,0.]],dtype=torch.float64)
        h=torch.tensor([[2.,1.],[-1.,4.],[3.,-2.]],dtype=p.dtype)
        s=torch.tensor([[.2],[1.],[-.7]],dtype=p.dtype)
        expected=h+s*(h@p.T)
        for order in [1,2,8,"exact"]:
            torch.testing.assert_close(action(h,p,s,order,chunk=2),expected,atol=1e-12,rtol=1e-12)

    def test_rotation_exposes_low_order_gain(self):
        p=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
        x=torch.eye(2,dtype=p.dtype);s=torch.ones((2,1),dtype=p.dtype)
        norms=[torch.linalg.matrix_norm(action(x,p,s,k),2).item() for k in [1,2,"exact"]]
        self.assertAlmostEqual(norms[0],2**.5,places=12)
        self.assertAlmostEqual(norms[1],1.25**.5,places=12)
        self.assertAlmostEqual(norms[2],1.,places=12)

    def test_context_restores_after_exception_and_keeps_weights(self):
        torch.manual_seed(0)
        model=VelocityNet(2,"ode_original",width=8,depth=1,gate_hidden=8,orders=2)
        # Avoid a zero output head concealing a failed intervention.
        torch.nn.init.normal_(model.output.weight)
        x=torch.randn(6,2);t=torch.rand(6,1)
        before={k:v.clone() for k,v in model.state_dict().items()}
        native=model(x,t).detach()
        with torch.no_grad(), evaluation_order(model,2):
            torch.testing.assert_close(model(x,t),native,atol=2e-6,rtol=2e-6)
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            with torch.no_grad(), evaluation_order(model,"exact"):
                self.assertTrue(torch.isfinite(model(x,t)).all())
                raise RuntimeError("intentional")
        torch.testing.assert_close(model(x,t),native,atol=0,rtol=0)
        for k,v in model.state_dict().items():
            torch.testing.assert_close(v,before[k],atol=0,rtol=0)

    def test_geometry_controls(self):
        matrices=[controlled_family(t) for t in [0.,.3,.7853981633974483]]
        for p in matrices:
            self.assertAlmostEqual(torch.linalg.matrix_norm(p,2).item(),1.,places=12)
            torch.testing.assert_close(torch.linalg.eigvals(p).real,
                                       torch.full((3,),-.25,dtype=p.dtype),atol=1e-12,rtol=0)
        # The controls allow different finite-time gains.
        gains=[torch.linalg.matrix_norm(torch.matrix_exp(p),2).item() for p in matrices]
        self.assertGreater(max(gains)-min(gains),1e-3)


if __name__ == "__main__":
    unittest.main()
