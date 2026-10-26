import unittest

import torch
from torch import nn
from torch.func import functional_call

from synthetic.conservative_readout import ReadoutNet, MatrixCounter, build
from synthetic.recurrent_depth import build as parent_build


def nonzero_head(model):
    with torch.no_grad():
        w=model.raw.output.weight
        w.copy_(torch.linspace(-.2,.3,w.numel(),dtype=w.dtype).reshape_as(w))


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.map=nn.Linear(2,2,dtype=torch.float64)
        with torch.no_grad():
            self.map.weight.copy_(torch.tensor([[.4,1.2],[-.3,.8]],dtype=torch.float64))
            self.map.bias.copy_(torch.tensor([.3,-.5],dtype=torch.float64))

    def forward(self,x,t):
        return self.map(x)


class ConservativeReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_affine_formula_and_actual_matrix_operation_count(self):
        raw=Affine()
        model=ReadoutNet(raw,2,'gradient')
        x=torch.tensor([[.2,-.5],[1.,.7]],dtype=torch.float64)
        t=torch.full((2,1),.4,dtype=torch.float64)
        a=raw.map.weight
        expected=x@((a+a.T)/2).T+raw.map.bias/2
        torch.testing.assert_close(model(x,t),expected,rtol=1e-13,atol=1e-13)
        with torch.no_grad(), MatrixCounter() as counter:
            model(x,t)
        self.assertEqual(counter.macs,16)  # 2 examples,2x2 forward and input reverse.

    def test_native_functions_states_and_initial_pairing(self):
        x=torch.tensor([[.2,-.5],[1.,.7]])
        t=torch.full((2,1),.4)
        for family in ('ode_k2','swiglu128','swiglu192'):
            direct=build(2,family,'direct')
            gradient=build(2,family,'gradient')
            for name,value in direct.state_dict().items():
                self.assertTrue(torch.equal(value,gradient.state_dict()[name]))
            parent=parent_build(2,family,'native')
            self.assertTrue(torch.equal(direct(x,t),parent(x,t)))
            self.assertTrue(torch.equal(direct(x,t),gradient(x,t)))

    def test_complete_jacobian_symmetry_and_independent_vjp_formula(self):
        x=torch.tensor([[.2,-.5],[1.,.7]],dtype=torch.float64)
        t=torch.full((2,1),.4,dtype=torch.float64)
        for family in ('ode_k2','swiglu128','swiglu192'):
            model=build(2,family,'gradient').double()
            nonzero_head(model)
            value=model(x,t)
            jac=torch.autograd.functional.jacobian(lambda z:model.raw(z,t),x)
            diagonal=torch.stack([jac[i,:,i,:] for i in range(len(x))])
            expected=.5*(model.raw(x,t)+torch.einsum('bij,bi->bj',diagonal,x))
            torch.testing.assert_close(value,expected,rtol=1e-11,atol=1e-12)
            derivative=torch.autograd.functional.jacobian(lambda z:model(z,t),x)
            for i in range(len(x)):
                local=derivative[i,:,i,:]
                torch.testing.assert_close(local,local.T,rtol=1e-11,atol=1e-12)
            torch.testing.assert_close(derivative[0,:,1,:],torch.zeros(2,2,dtype=x.dtype),atol=0,rtol=0)

    def test_mixed_input_parameter_derivatives_include_p_and_clock(self):
        model=build(2,'ode_k2','gradient').double()
        nonzero_head(model)
        x=torch.tensor([[.2,-.5]],dtype=torch.float64)
        t=torch.full((1,1),.4,dtype=torch.float64)
        keys=('raw.blocks.0.ffn.ode1.proj.weight','raw.blocks.0.ffn.ode1.t_func.2.weight')
        parameters=dict(model.named_parameters())
        values=tuple(parameters[k].detach().clone().requires_grad_(True) for k in keys)
        def evaluate(*updates):
            return functional_call(model,dict(zip(keys,updates)),(x,t))
        self.assertTrue(torch.autograd.gradcheck(evaluate,values,fast_mode=True))

    def test_no_grad_and_batch_evaluation_agree_without_mutating_input(self):
        model=build(2,'ode_k2','gradient').double().eval()
        nonzero_head(model)
        x=torch.tensor([[.2,-.5],[1.,.7]],dtype=torch.float64)
        t=torch.full((2,1),.4,dtype=torch.float64)
        reference=model(x,t)
        with torch.no_grad():
            value=model(x,t)
            separate=torch.cat([model(x[i:i+1],t[i:i+1]) for i in range(2)])
        self.assertFalse(value.requires_grad)
        self.assertFalse(x.requires_grad)
        torch.testing.assert_close(value,reference,rtol=0,atol=0)
        torch.testing.assert_close(separate,reference,rtol=1e-11,atol=1e-12)

    def test_actual_training_reaches_original_projection_and_controller(self):
        model=build(2,'ode_k2','gradient')
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        p=model.raw.blocks[0].ffn.ode1.proj.weight
        gate=model.raw.blocks[0].ffn.ode1.t_func[2].weight
        old_p,old_gate=p.detach().clone(),gate.detach().clone()
        x=torch.tensor([[.2,-.5],[1.,.7]])
        t=torch.full((2,1),.4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss=(model(x,t)-2*x).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        self.assertFalse(torch.equal(p,old_p))
        self.assertFalse(torch.equal(gate,old_gate))


if __name__=='__main__':
    unittest.main()
