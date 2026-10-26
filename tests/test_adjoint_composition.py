"""Literal residual-path and counterexample checks for the composition study."""
import unittest
import torch

from synthetic.adjoint_composition import AdjointComposition
from synthetic.conservative_readout import ReadoutNet
from synthetic.models import VelocityNet,OriginalODELayer


def enumerate_paths(raw,x,t):
    context=raw.time(t)
    hidden=raw.input(x)
    differences=[]
    width=raw.width
    for block in raw.blocks:
        output=block(hidden,context)
        jac=torch.stack([torch.autograd.grad(output[:,i].sum(),hidden,
            create_graph=True,retain_graph=True)[0] for i in range(width)],1)
        differences.append(jac-torch.eye(width,dtype=x.dtype).unsqueeze(0))
        hidden=output
    value=raw.output(raw.norm(hidden))
    seed=torch.autograd.grad(value,hidden,grad_outputs=x,create_graph=True,retain_graph=True)[0]
    components=[torch.zeros_like(x) for _ in range(4)]
    for mask in range(8):
        q=seed
        for index in (2,1,0):
            if mask & (1<<index):q=torch.einsum('bij,bi->bj',differences[index],q)
        v=.5*q@raw.input.weight
        degree=mask.bit_count()
        components[degree]=components[degree]+v
    components[0]=components[0]+.5*value
    return torch.stack(components,1)


class AdjointCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_eight_literal_paths_and_derivatives(self):
        for kind in ('ode_original','swiglu'):
            torch.manual_seed(0)
            raw=VelocityNet(2,kind,width=4,depth=3,gate_hidden=3,ffn_hidden=6,orders=2).double()
            with torch.no_grad():raw.output.weight.copy_(torch.linspace(-.2,.3,8).reshape(2,4))
            model=AdjointComposition(raw)
            x=torch.tensor([[-.7,.3],[.2,.8]],dtype=torch.float64,requires_grad=True)
            t=torch.tensor([[.23],[.71]],dtype=torch.float64,requires_grad=True)
            actual=model.components(x,t)
            expected=enumerate_paths(raw,x,t)
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
            for degree in range(4):
                args=(x,t,*raw.parameters())
                ga=torch.autograd.grad(actual[:,degree].square().sum(),args,allow_unused=True,retain_graph=True)
                gb=torch.autograd.grad(expected[:,degree].square().sum(),args,allow_unused=True,retain_graph=True)
                for a,b,v in zip(ga,gb,args):
                    aa=torch.zeros_like(v) if a is None else a
                    bb=torch.zeros_like(v) if b is None else b
                    torch.testing.assert_close(aa,bb,rtol=1e-8,atol=1e-10)
            reference=ReadoutNet(raw,2,'gradient')(x,t)
            torch.testing.assert_close(actual.sum(1),reference,rtol=1e-10,atol=1e-11)

    def test_local_gradient_maps_and_flows_need_not_compose_to_gradient(self):
        a=torch.diag(torch.tensor([2.,1.],dtype=torch.float64))
        b=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64)
        for matrix in (a,b):
            values,basis=torch.linalg.eigh(matrix)
            generator=(basis*values.log())@basis.T
            torch.testing.assert_close(generator,generator.T,rtol=0,atol=1e-14)
            torch.testing.assert_close(torch.matrix_exp(generator),matrix,rtol=1e-12,atol=1e-13)
        composed=b@a
        self.assertGreater(float((composed-composed.T).norm()),1.)
        torch.testing.assert_close(composed-composed.T,b@a-a@b,rtol=0,atol=0)

    def test_symmetric_generator_does_not_remove_clock_skew(self):
        layer=OriginalODELayer(2,2,orders=2).double()
        with torch.no_grad():
            layer.proj.weight.copy_(torch.diag(torch.tensor([.5,1.],dtype=torch.float64)))
            layer.t_func[0].weight.copy_(torch.eye(2,dtype=torch.float64));layer.t_func[0].bias.zero_()
            layer.t_func[2].weight.copy_(torch.tensor([[1.,0.]],dtype=torch.float64));layer.t_func[2].bias.zero_()
        x=torch.tensor([[1.,2.]],dtype=torch.float64,requires_grad=True)
        y=layer(x)
        jac=torch.stack([torch.autograd.grad(y[:,i].sum(),x,retain_graph=True)[0] for i in range(2)],1)[0]
        self.assertTrue(torch.equal(layer.proj.weight,layer.proj.weight.T))
        self.assertGreater(float((jac-jac.T).norm()),.01)
        self.assertTrue(.2<float(layer.step_size(x).detach())<1.)

    def test_inference_and_batch_independence(self):
        torch.manual_seed(0)
        raw=VelocityNet(2,'ode_original',width=4,depth=3,gate_hidden=3,orders=2).double()
        with torch.no_grad():raw.output.weight.copy_(torch.linspace(-.2,.3,8).reshape(2,4))
        model=AdjointComposition(raw)
        x=torch.tensor([[-.7,.3],[.2,.8]],dtype=torch.float64);t=torch.tensor([[.23],[.71]],dtype=torch.float64)
        with torch.inference_mode():
            batch=model.components(x,t)
            individual=torch.cat([model.components(x[i:i+1],t[i:i+1]) for i in range(2)])
        torch.testing.assert_close(batch,individual,rtol=1e-10,atol=1e-11)


if __name__=='__main__':unittest.main()
