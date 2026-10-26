import unittest

import torch
from torch import nn
from torch.func import functional_call

from synthetic.conservative_readout import ReadoutNet,build as parent_build
from synthetic.directional_readout import DirectionalReadout,build
from synthetic.differentiable_layernorm import DifferentiableLayerNorm,_LayerNorm


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

    def forward(self,x,t):return self.map(x)


class Cubic(nn.Module):
    def __init__(self):
        super().__init__();self.coefficient=nn.Parameter(torch.tensor(1.3,dtype=torch.float64))
    def forward(self,x,t):return self.coefficient*x**3


class DirectionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_affine_orientation_identity_and_nonzero_curl(self):
        raw=Affine();a=ReadoutNet(raw,2,'gradient');b=DirectionalReadout(raw,2)
        x=torch.tensor([[.2,-.5],[1.,.7]],dtype=torch.float64)
        t=torch.full((2,1),.4,dtype=torch.float64)
        expected=x@raw.map.weight.T+raw.map.bias/2
        torch.testing.assert_close(b(x,t),expected,rtol=1e-13,atol=1e-13)
        difference=.5*x@(raw.map.weight-raw.map.weight.T)
        torch.testing.assert_close(a(x,t)-b(x,t),difference,rtol=1e-13,atol=1e-13)
        jac=torch.autograd.functional.jacobian(lambda z:b(z,t),x)[0,:,0,:]
        torch.testing.assert_close(jac,raw.map.weight,rtol=1e-13,atol=1e-13)
        self.assertGreater(float((jac-jac.T).norm()),1.)

    def test_layernorm_native_values_and_complete_derivatives(self):
        native=nn.LayerNorm(4,eps=.003,dtype=torch.float64)
        with torch.no_grad():
            native.weight.copy_(torch.tensor([.7,1.1,-.3,.9],dtype=torch.float64))
            native.bias.copy_(torch.tensor([.2,-.1,.3,.1],dtype=torch.float64))
        layer=DifferentiableLayerNorm.from_native(native)
        x=torch.tensor([[.2,-.5,.4,1.2]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.equal(native(x),layer(x)))
        torch.testing.assert_close(torch.autograd.functional.jacobian(native,x),
            torch.autograd.functional.jacobian(layer,x),rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.autograd.gradgradcheck(layer,(x,),fast_mode=True))
        values=(x,native.weight.detach().clone().requires_grad_(True),
                native.bias.detach().clone().requires_grad_(True))
        self.assertTrue(torch.autograd.gradcheck(lambda a,w,b:_LayerNorm.apply(a,w,b,.003),values,fast_mode=True))
        self.assertTrue(torch.autograd.gradgradcheck(lambda a,w,b:_LayerNorm.apply(a,w,b,.003),values,fast_mode=True))
        def jvp(z):
            value,tangent=torch.func.jvp(layer,(z,),(z,))
            return .5*(value+tangent)
        self.assertTrue(torch.autograd.gradcheck(jvp,(x,),fast_mode=True))

    def test_input_dependent_tangent_is_not_detached(self):
        model=DirectionalReadout(Cubic(),2)
        x=torch.tensor([[.2,-.5]],dtype=torch.float64,requires_grad=True)
        t=torch.full((1,1),.4,dtype=torch.float64)
        torch.testing.assert_close(model(x,t),2*model.raw.coefficient*x**3,rtol=1e-13,atol=1e-13)
        jac=torch.autograd.functional.jacobian(lambda z:model(z,t),x)[0,:,0,:]
        torch.testing.assert_close(jac,torch.diag(6*model.raw.coefficient*x[0]**2),rtol=1e-13,atol=1e-13)

    def test_complete_actual_model_jvp_matches_raw_jacobian_rows(self):
        x=torch.tensor([[.2,-.5],[1.,.7]],dtype=torch.float64)
        t=torch.full((2,1),.4,dtype=torch.float64)
        for family in ('ode_k2','swiglu128','swiglu192'):
            model=build(2,family,'directional').double();nonzero_head(model)
            jac=torch.autograd.functional.jacobian(lambda z:model.raw(z,t),x)
            blocks=torch.stack([jac[i,:,i,:] for i in range(2)])
            expected=.5*(model.raw(x,t)+torch.einsum('bij,bj->bi',blocks,x))
            torch.testing.assert_close(model(x,t),expected,rtol=1e-11,atol=1e-12)

    def test_mixed_parameter_and_input_derivatives(self):
        model=build(2,'ode_k2','directional').double();nonzero_head(model)
        x=torch.tensor([[.2,-.5]],dtype=torch.float64,requires_grad=True)
        t=torch.full((1,1),.4,dtype=torch.float64)
        keys=('raw.blocks.0.ffn.ode1.proj.weight','raw.blocks.0.ffn.ode1.t_func.2.weight')
        parameters=dict(model.named_parameters())
        values=tuple(parameters[k].detach().clone().requires_grad_(True) for k in keys)
        self.assertTrue(torch.autograd.gradcheck(lambda z:model(z,t),(x,),fast_mode=True))
        self.assertTrue(torch.autograd.gradcheck(lambda *updates:functional_call(model,dict(zip(keys,updates)),(x,t)),values,fast_mode=True))

    def test_parent_initial_states_rng_and_native_training_updates(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            model=build(2,family,'directional');rng=torch.get_rng_state().clone()
            reference=parent_build(2,family,'gradient')
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,reference.state_dict()[name]))
        model=build(2,'ode_k2','directional')
        p=model.raw.blocks[0].ffn.ode1.proj.weight
        g=model.raw.blocks[0].ffn.ode1.t_func[2].weight
        old_p,old_g=p.detach().clone(),g.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        x=torch.tensor([[.2,-.5],[1.,.7]]);t=torch.full((2,1),.4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            (model(x,t)-2*x).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        self.assertFalse(torch.equal(old_p,p));self.assertFalse(torch.equal(old_g,g))

    def test_no_grad_batch_independence_and_inference_mode_boundary(self):
        model=build(2,'ode_k2','directional').double();nonzero_head(model)
        x=torch.tensor([[.2,-.5],[1.,.7]],dtype=torch.float64)
        t=torch.full((2,1),.4,dtype=torch.float64)
        expected=model(x,t)
        with torch.no_grad():
            value=model(x,t)
            separate=torch.cat([model(x[i:i+1],t[i:i+1]) for i in range(2)])
        torch.testing.assert_close(value,expected,rtol=0,atol=0)
        torch.testing.assert_close(separate,expected,rtol=1e-11,atol=1e-12)
        self.assertFalse(value.requires_grad);self.assertFalse(x.requires_grad)
        with torch.inference_mode():
            with self.assertRaises(RuntimeError):model(x,t)


if __name__=='__main__':unittest.main()
