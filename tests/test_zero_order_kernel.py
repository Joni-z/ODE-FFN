import copy
import unittest

import torch

from synthetic.models import OriginalODELayer


def tangent_kernel(layer,x):
    prediction=layer(x[None])[0]
    parameters=tuple(layer.parameters())
    rows=[]
    for axis in range(len(x)):
        gradients=torch.autograd.grad(prediction[axis],parameters,retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in gradients]))
    response=torch.stack(rows)
    return response@response.T


class ZeroOrderKernelTests(unittest.TestCase):
    def test_same_initial_functions_can_raise_or_lower_full_layer_kernel(self):
        torch.manual_seed(0);torch.set_num_threads(1)
        x=torch.tensor([1.,2.],dtype=torch.float64)
        for alpha,sign in ((0.,1),(-.75,-1)):
            parent=OriginalODELayer(2,4,orders=1).double()
            center=parent.shift+parent.scale/2
            with torch.no_grad():
                parent.t_func[-1].weight.zero_();parent.t_func[-1].bias.zero_()
                parent.proj.weight.copy_(torch.eye(2,dtype=torch.float64)*alpha/center)
            increment=copy.deepcopy(parent);increment.solve_return='increment'
            with torch.no_grad():increment.proj.weight.add_(torch.eye(2,dtype=torch.float64)/center)
            torch.testing.assert_close(parent(x[None]),increment(x[None]),atol=1e-15,rtol=1e-14)
            difference=tangent_kernel(increment,x)-tangent_kernel(parent,x)
            eigenvalues=torch.linalg.eigvalsh(sign*difference)
            self.assertGreater(float(eigenvalues[-1]),1e-4)
            self.assertGreaterEqual(float(eigenvalues[0]),-1e-14)


if __name__=='__main__':
    unittest.main()
