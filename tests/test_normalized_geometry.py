"""Exact normalized-state identities and original-ODE spectral nonidentifiability."""
import unittest

import torch
from torch import nn
from torch.func import functional_call

from synthetic.normalized_geometry import GeometryNet,decompose
from synthetic.recurrent_depth import build


def nonzero_head(model):
    with torch.no_grad():
        model.raw.output.weight.copy_(torch.linspace(-.2,.2,model.raw.output.weight.numel(),
            dtype=model.raw.output.weight.dtype).reshape_as(model.raw.output.weight))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_decomposition_and_exact_centered_residual_update(self):
        x=torch.tensor([[1.,-2.,4.,3.],[.2,.7,-.4,1.]],dtype=torch.float64)
        f=torch.tensor([[.4,.9,-.3,.7],[1.,-.5,.7,.1]],dtype=torch.float64)
        c,u,r,q,beta=decompose(x,f)
        torch.testing.assert_close(f,u+r+q,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(q.mean(-1),torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close((q*c).sum(-1),torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close(f.square().sum(-1),sum(z.square().sum(-1) for z in (u,r,q)),rtol=1e-14,atol=1e-14)
        a=.3;next_state=x+a*f
        torch.testing.assert_close(next_state-next_state.mean(-1,keepdim=True),(1+a*beta)*c+a*q,rtol=1e-14,atol=1e-14)

    def test_layernorm_derivative_includes_finite_epsilon(self):
        x=torch.tensor([.2,-.6,.9,1.2],dtype=torch.float64)
        layer=nn.LayerNorm(4,eps=.02,dtype=torch.float64)
        with torch.no_grad():layer.weight.copy_(torch.tensor([.8,1.1,.7,1.3]))
        c=x-x.mean();sigma=torch.sqrt(c.square().mean()+layer.eps)
        projection=torch.eye(4,dtype=torch.float64)-torch.ones(4,4,dtype=torch.float64)/4
        expected=torch.diag(layer.weight)/sigma@(projection-torch.outer(c,c)/(4*sigma.square()))
        actual=torch.autograd.functional.jacobian(layer,x)
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(actual@torch.ones_like(x),torch.zeros_like(x),atol=1e-14,rtol=0)
        torch.testing.assert_close(actual@c,layer.weight*c*layer.eps/sigma**3,rtol=1e-13,atol=1e-13)

    def test_native_wrapper_and_uniform_removal_preserve_full_model_and_derivatives(self):
        x=torch.tensor([[.2,-.6],[1.,.4]],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[.25],[.75]],dtype=torch.float64)
        for family in ('ode_k2','swiglu192','swiglu128'):
            parent=build(2,family,'native').double();nonzero_head(parent)
            native=GeometryNet(parent);centered=GeometryNet(parent,'drop_uniform')
            self.assertTrue(torch.equal(native(x,t),parent(x,t)))
            torch.testing.assert_close(native(x,t),centered(x,t),rtol=1e-12,atol=1e-12)
            a=torch.autograd.functional.jacobian(lambda z:native(z,t),x)
            b=torch.autograd.functional.jacobian(lambda z:centered(z,t),x)
            torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)

    def test_original_k2_second_generator_can_have_arbitrary_invisible_mean_eigenvalue(self):
        parent=build(2,'ode_k2','native').double();nonzero_head(parent)
        x=torch.tensor([[.2,-.6],[1.,.4]],dtype=torch.float64)
        t=torch.tensor([[.25],[.75]],dtype=torch.float64)
        names=[name+'.proj.weight' for name,_ in parent.ode_layers() if name.endswith('ode2')]
        projector=torch.ones(64,64,dtype=torch.float64)/64
        output=[]
        for eigenvalue in (0.,1.,10.):
            updates={name:eigenvalue*projector for name in names}
            output.append(functional_call(parent,updates,(x,t)))
        for value in output[1:]:torch.testing.assert_close(value,output[0],rtol=2e-11,atol=2e-11)
        scalar=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        value=functional_call(parent,{name:scalar*projector for name in names},(x,t))
        gradient=torch.autograd.grad(value.sum(),scalar)[0]
        self.assertLess(abs(float(gradient)),1e-12)

    def test_projected_variants_have_actual_input_derivatives(self):
        parent=build(2,'ode_k2','native').double();nonzero_head(parent)
        x=torch.tensor([[.2,-.6]],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[.4]],dtype=torch.float64)
        for variant in ('drop_radial','drop_tangent'):
            model=GeometryNet(parent,variant)
            self.assertTrue(torch.autograd.gradcheck(lambda z:model(z,t),(x,),fast_mode=True))

    def test_zero_centered_state_has_declared_finite_decomposition(self):
        x=torch.ones(2,4,dtype=torch.float64)
        f=torch.arange(8,dtype=torch.float64).reshape(2,4)
        _,u,r,q,beta=decompose(x,f)
        self.assertTrue(torch.equal(r,torch.zeros_like(r)))
        self.assertTrue(torch.isfinite(beta).all())
        torch.testing.assert_close(q,f-u)

    def test_actual_trace_preserves_outputs_and_identifies_used_branch(self):
        parent=build(2,'ode_k2','native').double();nonzero_head(parent)
        x=torch.tensor([[.2,-.6],[1.,.4]],dtype=torch.float64)
        t=torch.tensor([[.25],[.75]],dtype=torch.float64)
        for variant in ('native','drop_uniform','drop_radial','drop_tangent'):
            model=GeometryNet(parent,variant);trace=[]
            value=model(x,t,capture=trace)
            self.assertTrue(torch.equal(value,model(x,t)))
            self.assertEqual(len(trace),3)
            for row in trace:
                hidden=row['hidden'];step=row['step'];branch=row['used_branch']
                torch.testing.assert_close(row['next_hidden'],hidden+step*branch,rtol=0,atol=0)
                if variant=='drop_radial':
                    c=row['centered'];next_c=row['next_hidden']-row['next_hidden'].mean(-1,keepdim=True)
                    torch.testing.assert_close(next_c.square().sum(-1),c.square().sum(-1)+step**2*row['tangent'].square().sum(-1),rtol=1e-13,atol=1e-13)


if __name__=='__main__':unittest.main()
