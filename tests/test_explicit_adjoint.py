"""Complete derivative and optimizer checks for the explicit primal/adjoint."""
import copy
import unittest

import torch

from synthetic.conservative_readout import build
from synthetic.explicit_adjoint import ExplicitAdjointNet,ode_primal,ode_adjoint
from synthetic.models import OriginalODELayer


def pair(dim,family):
    reference=build(dim,family,'gradient').double()
    with torch.no_grad():
        reference.raw.output.weight.copy_(torch.linspace(-.11,.13,reference.raw.output.weight.numel()).reshape_as(reference.raw.output.weight))
        reference.raw.output.bias.copy_(torch.linspace(-.07,.09,dim))
    explicit=ExplicitAdjointNet(copy.deepcopy(reference.raw))
    return reference,explicit


def jacobian(value,x):
    return torch.stack([torch.autograd.grad(value[:,j].sum(),x,retain_graph=True)[0]
                        for j in range(value.shape[-1])],1)


class ExplicitAdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_layer_values_and_full_adjoint(self):
        torch.manual_seed(0)
        layer=OriginalODELayer(5,7).double()
        h=torch.randn(4,5,dtype=torch.float64,requires_grad=True)
        q=torch.randn_like(h)
        value,cache=ode_primal(layer,h)
        torch.testing.assert_close(value,layer(h),rtol=0,atol=0)
        actual=ode_adjoint(cache,q)
        expected=torch.autograd.grad(layer(h),h,grad_outputs=q,create_graph=True)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
        ga=torch.autograd.grad(actual.square().sum(),(h,*layer.parameters()))
        gb=torch.autograd.grad(expected.square().sum(),(h,*layer.parameters()))
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)

    def test_complete_values_and_input_jacobians(self):
        for dim in (2,16):
            for family in ('ode_k2','swiglu128','swiglu192'):
                with self.subTest(dim=dim,family=family):
                    a,b=pair(dim,family)
                    x=torch.linspace(-.7,.9,3*dim,dtype=torch.float64).reshape(3,dim).requires_grad_()
                    t=torch.tensor([[.13],[.47],[.83]],dtype=torch.float64)
                    av,bv=a(x,t),b(x,t)
                    torch.testing.assert_close(av,bv,rtol=1e-10,atol=1e-11)
                    torch.testing.assert_close(jacobian(av,x),jacobian(bv,x),rtol=1e-9,atol=1e-10)

    def test_all_parameter_and_time_loss_gradients(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            a,b=pair(3,family)
            x=torch.tensor([[-.4,.7,.2],[.6,-.1,.8]],dtype=torch.float64,requires_grad=True)
            t=torch.tensor([[.2],[.6]],dtype=torch.float64,requires_grad=True)
            target=x.detach().square()+.17
            av,bv=a(x,t),b(x,t)
            ga=torch.autograd.grad((av-target).square().mean(),(x,t,*a.parameters()))
            gb=torch.autograd.grad((bv-target).square().mean(),(x,t,*b.parameters()))
            self.assertEqual([n for n,_ in a.named_parameters()],[n for n,_ in b.named_parameters()])
            for k,(aa,bb) in enumerate(zip(ga,gb)):
                with self.subTest(family=family,gradient=k):
                    torch.testing.assert_close(aa,bb,rtol=1e-8,atol=1e-10)

    def test_actual_adamw_update(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            a,b=pair(2,family)
            x=torch.tensor([[-.4,.7],[.6,-.1]],dtype=torch.float64)
            t=torch.tensor([[.2],[.6]],dtype=torch.float64)
            for model in (a,b):
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
                optimizer.zero_grad()
                (model(x,t)-x.square()).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            for (name,aa),(_,bb) in zip(a.named_parameters(),b.named_parameters()):
                with self.subTest(family=family,parameter=name):
                    torch.testing.assert_close(aa,bb,rtol=1e-9,atol=1e-10)

    def test_no_grad_inference_and_batch_independence(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            a,b=pair(2,family)
            x=torch.tensor([[-.4,.7],[.6,-.1]],dtype=torch.float64)
            t=torch.tensor([[.2],[.6]],dtype=torch.float64)
            expected=a(x,t).detach()
            for context in (torch.no_grad(),torch.inference_mode()):
                with context:
                    result=b(x,t)
                    torch.testing.assert_close(result,expected,rtol=1e-10,atol=1e-11)
                    single=torch.cat([b(x[i:i+1],t[i:i+1]) for i in range(len(x))])
                    torch.testing.assert_close(result,single,rtol=1e-10,atol=1e-11)
                    self.assertFalse(result.requires_grad)

    def test_gate_omission_is_a_differentiable_vector_function(self):
        _,model=pair(2,'ode_k2')
        model.include_clock=False
        x=torch.tensor([[-.4,.7],[.6,-.1]],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[.2],[.6]],dtype=torch.float64)
        direction=torch.tensor([[.3,-.2],[.5,.1]],dtype=torch.float64)
        value=model(x,t)
        derivatives=jacobian(value,x)
        actual=torch.einsum('bij,bj->bi',derivatives,direction)
        with torch.no_grad():
            eps=1e-5
            finite=(model(x+eps*direction,t)-model(x-eps*direction,t))/(2*eps)
        torch.testing.assert_close(actual,finite,rtol=1e-6,atol=1e-8)
        self.assertTrue(all(torch.isfinite(v).all() for v in torch.autograd.grad(value.square().sum(),tuple(model.parameters()))))

    def test_rejects_changed_local_dynamics(self):
        layer=OriginalODELayer(3,4,orders=1).double()
        with self.assertRaisesRegex(ValueError,'scalar K2'):
            ode_primal(layer,torch.ones(2,3,dtype=torch.float64))


if __name__=='__main__':unittest.main()
