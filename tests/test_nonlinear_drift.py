"""Nonlinear ODE Taylor coefficients, increment semantics and matched budgets."""
import unittest
import torch
from synthetic.models import NonlinearDriftFFN,VelocityNet
from synthetic.diagnostics import collect
from scripts.run_architecture_study import macs


class NonlinearDriftTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1);torch.manual_seed(0)

    def test_frozen_controller_second_term_is_exact_field_jvp(self) -> None:
        block=NonlinearDriftFFN(4,3,4,orders=2).double()
        x=torch.randn(4,dtype=torch.float64,requires_grad=True)
        duration,first,second=block.terms(x[None])
        d=duration[0].detach()
        field=lambda y:d*block.proj2(torch.nn.functional.silu(block.proj1(y)))
        jac=torch.autograd.functional.jacobian(field,x)
        torch.testing.assert_close(second[0],.5*jac@first[0],atol=1e-14,rtol=1e-12)

    def test_taylor_local_error_orders_against_rk4(self) -> None:
        block=NonlinearDriftFFN(3,3,3,orders=2).double()
        x=torch.tensor([[.7,-.6,.9]],dtype=torch.float64)
        s,first,second=block.terms(x)
        s=s.detach()
        def field(y):return s*block.proj2(torch.nn.functional.silu(block.proj1(y)))
        errors=[]
        with torch.no_grad():
            for horizon in [.2,.1,.05]:
                y=x.clone();dt=horizon/128
                for _ in range(128):
                    a=field(y);b=field(y+.5*dt*a);c=field(y+.5*dt*b);d=field(y+dt*c)
                    y=y+dt*(a+2*b+2*c+d)/6
                errors.append([float((y-x-horizon*first).norm()),
                    float((y-x-horizon*first-horizon**2*second).norm())])
        self.assertGreater(errors[0][0]/errors[1][0],3.5)
        self.assertLess(errors[0][0]/errors[1][0],4.5)
        self.assertGreater(errors[0][1]/errors[1][1],7.)
        self.assertLess(errors[0][1]/errors[1][1],9.)

    def test_map_increment_fixed_gate_and_zero_generator_limits(self) -> None:
        x=torch.randn(7,4,dtype=torch.float64)
        increment=NonlinearDriftFFN(4,3,4).double()
        mapping=NonlinearDriftFFN(4,3,4,return_mode='map').double()
        mapping.load_state_dict(increment.state_dict())
        torch.testing.assert_close(mapping(x),x+increment(x),atol=0,rtol=0)
        fixed=NonlinearDriftFFN(4,3,4,gate_source='fixed').double()
        duration=fixed.shift+fixed.scale/2
        torch.testing.assert_close(fixed(x),duration*fixed.proj2(torch.nn.functional.silu(fixed.proj1(x))),atol=0,rtol=0)
        with torch.no_grad():increment.proj2.weight.zero_()
        torch.testing.assert_close(increment(x),torch.zeros_like(x),atol=0,rtol=0)

    def test_initial_pairing_probes_gradients_and_linear_macs(self) -> None:
        torch.manual_seed(0);reference=VelocityNet(2,'ode_vector',orders=1,gate_hidden=32)
        old_rng=torch.get_rng_state()
        torch.manual_seed(0);candidate=VelocityNet(2,'ode_nonlinear',orders=1,gate_hidden=32)
        torch.testing.assert_close(old_rng,torch.get_rng_state(),atol=0,rtol=0)
        for name,value in candidate.state_dict().items():
            mapped=name.replace('.ffn.proj1.','.ffn.ode1.proj.').replace('.ffn.proj2.','.ffn.ode2.proj.').replace('.ffn.t_func.','.ffn.ode1.t_func.')
            torch.testing.assert_close(value,reference.state_dict()[mapped],atol=0,rtol=0)
        for index,block in enumerate(candidate.blocks):
            old=reference.blocks[index].ffn;new=block.ffn
            torch.testing.assert_close(new.proj1.weight,old.ode1.proj.weight,atol=0,rtol=0)
            torch.testing.assert_close(new.proj2.weight,old.ode2.proj.weight,atol=0,rtol=0)
            for key,value in new.t_func.state_dict().items():
                torch.testing.assert_close(value,old.ode1.t_func.state_dict()[key],atol=0,rtol=0)
        self.assertEqual(candidate.parameter_count(),69282)
        self.assertEqual(macs(candidate,2),67904)
        self.assertEqual(macs(candidate,2),macs(VelocityNet(2,'swiglu',ffn_hidden=96),2))
        with torch.no_grad():candidate.output.weight.normal_()
        x=torch.randn(16,2);t=torch.rand(16,1)
        rows=collect(candidate,x,t)
        self.assertEqual(len(rows),6)
        self.assertTrue(all(r['operator_type']=='nonlinear_drift_factor' for r in rows))
        candidate(x,t).square().mean().backward()
        for block in candidate.blocks:
            self.assertGreater(float(block.ffn.proj1.weight.grad.abs().sum()),0)
            self.assertGreater(float(block.ffn.proj2.weight.grad.abs().sum()),0)
            self.assertGreater(float(block.ffn.t_func[0].weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
