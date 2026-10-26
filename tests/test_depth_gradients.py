import math
import unittest
import torch
from synthetic.depth_gradients import (independent_copies,depth_basis,describe,projected,
    transferred_alignment,block_coordinates,loss_gradients,block_vectors,copy_block_vectors)
from synthetic.recurrent_potential import build


class DepthGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_orthogonal_modes_and_extremes(self):
        x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.float64)
        basis=depth_basis(x)
        torch.testing.assert_close(basis@basis.T,torch.eye(3,dtype=torch.float64),rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(projected(x,(0,1,2)),x,rtol=1e-13,atol=1e-13)
        self.assertAlmostEqual(describe(torch.ones(3,2,dtype=torch.float64))['retained'],1.)
        self.assertAlmostEqual(describe(torch.eye(3,dtype=torch.float64))['retained'],1/3)
        self.assertLess(describe(torch.tensor([[1.,0.],[0.,0.],[-1.,0.]],dtype=torch.float64))['retained'],1e-20)
        self.assertIsNone(describe(torch.zeros(3,2,dtype=torch.float64))['retained'])

    def test_projection_metric_and_transfer(self):
        a=torch.tensor([[1.,2.],[-2.,3.],[4.,-1.]],dtype=torch.float64)
        b=torch.tensor([[.2,1.],[-1.,2.],[2.,-.4]],dtype=torch.float64)
        p0=projected(a,(0,));p1=projected(a,(1,));p2=projected(a,(2,))
        torch.testing.assert_close(p0,torch.ones_like(a)*a.mean(0),rtol=1e-13,atol=1e-13)
        torch.testing.assert_close((p0*p1).sum(),a.new_zeros(()),rtol=0,atol=1e-12)
        torch.testing.assert_close((p0+p1+p2).square().sum(),a.square().sum(),rtol=1e-13,atol=1e-13)
        expected=float((b*(p0+p1)).sum()/((p0+p1).norm()*b.norm()))
        self.assertAlmostEqual(transferred_alignment(a,b,(0,1)),expected)

    def test_actual_matched_functions_and_all_gradient_sums(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            for readout in ('direct','gradient'):
                shared=build(3,family,'shared_'+readout).double()
                with torch.no_grad():
                    for parameter in shared.parameters():parameter.add_(torch.linspace(-.003,.005,parameter.numel()).reshape_as(parameter))
                clone=independent_copies(shared)
                x=torch.tensor([[-.2,.3,.7],[.4,-.6,.1]],dtype=torch.float64)
                t=torch.tensor([[.3],[.6]],dtype=torch.float64);target=x.square()+.1
                a,la,ga,ca=loss_gradients(shared,x,t,target)
                b,lb,gb,cb=loss_gradients(clone,x,t,target)
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
                self.assertAlmostEqual(la,lb)
                torch.testing.assert_close(ga[0],gb.sum(0),rtol=1e-7,atol=1e-9)
                for name in ca:torch.testing.assert_close(ca[name],cb[name],rtol=1e-7,atol=1e-9)
                names,sizes,groups=block_coordinates(shared,family)
                self.assertEqual(sum(sizes),ga.shape[-1])
                self.assertEqual(len(groups['all']),ga.shape[-1])

    def test_vector_restore_and_finite_direction(self):
        model=build(2,'ode_k2','shared_gradient').double()
        with torch.no_grad():model.raw.raw.output.weight.copy_(torch.linspace(-.1,.2,128).reshape(2,64))
        clone=independent_copies(model)
        x=torch.tensor([[-.2,.7],[.4,-.3]],dtype=torch.float64);t=torch.tensor([[.2],[.6]],dtype=torch.float64)
        target=x.square()
        _,_,g,_=loss_gradients(clone,x,t,target)
        direction=-projected(g,(0,1));direction/=direction.norm()
        original=block_vectors(clone).clone();eps=1e-5
        with torch.no_grad():
            copy_block_vectors(clone,original+eps*direction);plus=(clone(x,t)-target).square().mean()
            copy_block_vectors(clone,original-eps*direction);minus=(clone(x,t)-target).square().mean()
            copy_block_vectors(clone,original)
        torch.testing.assert_close((plus-minus)/(2*eps),(g*direction).sum(),rtol=1e-5,atol=1e-8)
        torch.testing.assert_close(block_vectors(clone),original,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
