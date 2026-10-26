"""Analytical equal-budget and first-order controls for temporal interventions."""
import unittest

import torch

from scripts.check_time_operator import directions


class TimeOperatorTests(unittest.TestCase):
    def test_orthogonal_time_components_and_equal_displacement_budget(self):
        phi=torch.tensor([-2.,-1.,0.,1.,2.],dtype=torch.float64)/(2.**.5)
        common=torch.tensor([1.,2.,-1.],dtype=torch.float64)
        temporal=torch.tensor([-3.,1.,2.],dtype=torch.float64)
        gradients=common+phi[:,None]*temporal
        variants,info=directions(gradients,phi)
        self.assertAlmostEqual(info['pooled_gradient_norm'],float(common.norm()),12)
        self.assertAlmostEqual(info['temporal_gradient_norm'],float(temporal.norm()),12)
        for value in variants.values():
            torch.testing.assert_close(value.square().sum(1).mean(),torch.tensor(.0001,dtype=torch.float64),rtol=1e-13,atol=1e-16)
        torch.testing.assert_close(variants['time_linear'].flip(0),variants['time_reverse'],rtol=0,atol=0)

    def test_exact_quadratic_loss_contains_declared_linear_term_and_curvature(self):
        phi=torch.tensor([-2.,-1.,0.,1.,2.],dtype=torch.float64)/(2.**.5)
        weight=torch.tensor([.5,-.3],dtype=torch.float64)
        targets=torch.tensor([[-1.,2.],[.4,.8],[1.,-.2],[-.6,-.5],[2.,1.]],dtype=torch.float64)
        gradient=weight-targets
        variants,info=directions(gradient,phi)
        baseline=.5*(weight-targets).square().sum(1).mean()
        for name,change in variants.items():
            after=.5*(weight+change-targets).square().sum(1).mean()
            curvature=.5*change.square().sum(1).mean()
            self.assertAlmostEqual(float(after-baseline-curvature),info['first_order'][name],13)

    def test_zero_pooled_gradient_is_indeterminate_instead_of_nan(self):
        phi=torch.tensor([-2.,-1.,0.,1.,2.],dtype=torch.float64)/(2.**.5)
        values,info=directions(torch.zeros(5,4,dtype=torch.float64),phi)
        self.assertTrue(info['zero_joint_gradient'])
        self.assertTrue(all(torch.equal(v,torch.zeros_like(v)) for v in values.values()))
        with self.assertRaises(ValueError):directions(torch.zeros(5,4),torch.ones(5))


if __name__=='__main__':unittest.main()
