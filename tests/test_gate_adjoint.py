import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from synthetic.conservative_readout import build as parent_build
from synthetic.explicit_adjoint import ffn_primal,ffn_adjoint,silu_prime
from synthetic.gate_adjoint import build,visit_diagnostics


class GateAdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_paired_initialization_and_rng(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            full=build(16,family,'gradient');rng_full=torch.get_rng_state().clone()
            omitted=build(16,family,'gate_omitted');rng_omitted=torch.get_rng_state().clone()
            self.assertTrue(torch.equal(rng_full,rng_omitted))
            for a,b in zip(full.state_dict().values(),omitted.state_dict().values()):self.assertTrue(torch.equal(a,b))
            for a,b in zip(full.buffers(),omitted.buffers()):self.assertTrue(torch.equal(a,b))
            x=torch.linspace(-1,1,32).reshape(2,16);t=torch.tensor([[.2],[.7]])
            torch.testing.assert_close(full(x,t),omitted(x,t),rtol=0,atol=0)

    def test_swiglu_omitted_gate_is_the_declared_branch(self):
        model=build(3,'swiglu128','gradient').double()
        module=model.raw.blocks[0].ffn
        # Local FFN width is64, use a separate actual hidden input.
        h=torch.linspace(-.7,.8,4*64,dtype=torch.float64).reshape(4,64)
        q=torch.cos(h)
        _,cache=ffn_primal(module,h)
        gate,value=module.w12(h).chunk(2,-1)
        hidden=F.linear(q,module.w3.weight.T)
        expected=F.linear(torch.cat((torch.zeros_like(hidden),hidden*F.silu(gate)),-1),module.w12.weight.T)
        actual=ffn_adjoint(cache,q,include_swiglu_gate=False)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        full=ffn_adjoint(cache,q)
        gate_part=F.linear(torch.cat((hidden*value*silu_prime(gate),torch.zeros_like(hidden)),-1),module.w12.weight.T)
        torch.testing.assert_close(full-actual,gate_part,rtol=1e-11,atol=1e-12)

    def test_full_omission_input_time_and_parameter_finite_differences(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            model=build(3,family,'gate_omitted').double()
            with torch.no_grad():model.raw.output.weight.copy_(torch.linspace(-.1,.13,192).reshape(3,64))
            x=torch.tensor([[-.3,.7,.1],[.4,-.6,.8]],dtype=torch.float64,requires_grad=True)
            t=torch.tensor([[.2],[.6]],dtype=torch.float64,requires_grad=True)
            parameter=(model.raw.blocks[1].ffn.ode1.t_func[0].weight if family=='ode_k2'
                       else model.raw.blocks[1].ffn.w12.weight)
            directions=[torch.cos(torch.arange(v.numel(),dtype=torch.float64)).reshape_as(v) for v in (x,t,parameter)]
            loss=model(x,t).square().mean()
            gradients=torch.autograd.grad(loss,(x,t,parameter))
            for index,(v,d,g) in enumerate(zip((x,t,parameter),directions,gradients)):
                # Time features reach1000 cycles/unit; resolve their curvature.
                eps=1e-7 if index==1 else 1e-5
                with torch.no_grad():
                    if index<2:
                        args=[x,t];args[index]=v+eps*d;plus=model(*args).square().mean()
                        args[index]=v-eps*d;minus=model(*args).square().mean()
                    else:
                        original=parameter.clone();parameter.copy_(original+eps*d);plus=model(x,t).square().mean()
                        parameter.copy_(original-eps*d);minus=model(x,t).square().mean();parameter.copy_(original)
                expected=(g*d).sum();actual=(plus-minus)/(2*eps)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-8)

    def test_actual_ode_gate_visits(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            np.savez(folder/'probe.npz',x=np.linspace(-1,1,64,dtype=np.float32).reshape(32,2),t=np.full((32,1),.4,dtype=np.float32))
            rows=visit_diagnostics(build(2,'ode_k2','gradient'),folder)
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['activation']['input_rms']>0 for r in rows))
            self.assertEqual(visit_diagnostics(build(2,'swiglu128','gate_omitted'),folder),[])

    def test_full_readout_matches_original(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            old=parent_build(3,family,'gradient').double()
            new=build(3,family,'gradient').double()
            with torch.no_grad():
                values=torch.linspace(-.1,.13,192).reshape(3,64)
                for model in (old,new):model.raw.output.weight.copy_(values)
            x=torch.tensor([[-.3,.7,.1],[.4,-.6,.8]],dtype=torch.float64)
            t=torch.tensor([[.2],[.6]],dtype=torch.float64)
            torch.testing.assert_close(old(x,t),new(x,t),rtol=1e-10,atol=1e-11)


if __name__=='__main__':unittest.main()
