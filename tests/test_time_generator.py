"""Dense operator, derivative, initialization and matrix-budget prototype controls."""
import math
import unittest

import torch

from synthetic.time_generator import build,TimeODELayer,time_coordinate,LowRankTime
from synthetic.recurrent_depth import build as parent_build,compute_budget


def nonzero_head(model):
    with torch.no_grad():
        model.raw.output.weight.copy_(torch.linspace(-.2,.2,model.raw.output.weight.numel(),
            dtype=model.raw.output.weight.dtype).reshape_as(model.raw.output.weight))


class TimeGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_none_control_retains_actual_default_optimization_trajectory(self):
        for family in ("ode_k2","swiglu192"):
            a,b=build(2,family,"none"),parent_build(2,family,"native")
            optimizers=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0) for m in (a,b)]
            x=torch.linspace(-1,1,20).reshape(10,2);t=torch.linspace(.1,.9,10)[:,None]
            for _ in range(3):
                for model,optimizer in zip((a,b),optimizers):
                    optimizer.zero_grad();loss=(model(x,t)+x).square().mean();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]))

    def test_zero_adapter_preserves_nonzero_head_functions_and_full_input_derivatives(self):
        x=torch.tensor([[.2,-.5],[.6,.8]],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[.25],[.75]],dtype=torch.float64)
        for family in ("ode_k2","swiglu192"):
            models=[build(2,family,mode).double() for mode in ("none","static","time")]
            for model in models:nonzero_head(model)
            values=[m(x,t) for m in models]
            for v in values[1:]:self.assertTrue(torch.equal(values[0],v))
            derivatives=[torch.autograd.functional.jacobian(lambda z:m(z,t),x) for m in models]
            for v in derivatives[1:]:torch.testing.assert_close(derivatives[0],v,rtol=0,atol=0)

    def test_effective_generator_is_used_in_both_taylor_powers_and_gradients(self):
        model=build(2,"ode_k2","time").double();layer=model.raw.blocks[0].ffn.ode1
        with torch.no_grad():layer.adapter.up.weight.fill_(.03)
        x=torch.linspace(-.5,.5,128,dtype=torch.float64).reshape(2,64).requires_grad_(True)
        t=torch.tensor([[.2],[.8]],dtype=torch.float64,requires_grad=True)
        matrix=layer.proj.weight[None]+time_coordinate(t)[:,:,None]*layer.adapter.matrix()[None]
        s=layer.step_size(x)
        first=(matrix@x[:,:,None]).squeeze(-1)*s
        second=(matrix@first[:,:,None]).squeeze(-1)*s/2
        expected=x+first+second;actual=layer(x,t)
        torch.testing.assert_close(actual,expected,rtol=2e-13,atol=2e-13)
        variables=(x,t,layer.proj.weight,layer.adapter.down.weight,layer.adapter.up.weight)
        a=torch.autograd.grad(actual.square().mean(),variables,retain_graph=True)
        b=torch.autograd.grad(expected.square().mean(),variables)
        for one,two in zip(a,b):torch.testing.assert_close(one,two,rtol=1e-11,atol=1e-13)

    def test_taylor_frechet_derivative_contains_both_noncommuting_terms(self):
        p=torch.tensor([[.2,.3],[-.1,.4]],dtype=torch.float64)
        delta=torch.tensor([[.1,-.5],[.3,-.2]],dtype=torch.float64)
        x=torch.tensor([.4,-.7],dtype=torch.float64);s=.6
        self.assertGreater(float((p@delta-delta@p).norm()),.01)
        scale=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        f=lambda z:x+s*(p+z*delta)@x+.5*s*s*(p+z*delta)@(p+z*delta)@x
        actual=torch.autograd.functional.jacobian(f,scale)
        expected=s*delta@x+.5*s*s*(p@delta+delta@p)@x
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)

    def test_extra_executed_macs_match_generic_and_parameters_reflect_power_reuse(self):
        values={}
        for family in ("ode_k2","swiglu192"):
            base=compute_budget(build(16,family,"none"))
            for mode in ("static","time"):
                added=compute_budget(build(16,family,mode))
                self.assertEqual(added['dense_forward_macs']-base['dense_forward_macs'],6144)
                self.assertEqual(added['parameters']-base['parameters'],3072 if family=='ode_k2' else 6144)
                values[family]=added
        self.assertLess(abs(values['ode_k2']['dense_forward_macs']/values['swiglu192']['dense_forward_macs']-1),.01)

    def test_initial_common_gradients_match_and_temporal_adapter_gradient_is_scaled(self):
        x=torch.tensor([[.2,-.5],[.6,.8]],dtype=torch.float64)
        t=torch.full((2,1),.25,dtype=torch.float64)
        coefficient=float(time_coordinate(t)[0,0])
        for family in ("ode_k2","swiglu192"):
            models=[build(2,family,mode).double() for mode in ("none","static","time")]
            for m in models:nonzero_head(m);m(x,t).square().mean().backward()
            reference=dict(models[0].named_parameters())
            for model in models[1:]:
                for name,parameter in model.named_parameters():
                    if name in reference:torch.testing.assert_close(parameter.grad,reference[name].grad,rtol=1e-11,atol=1e-13)
            static=dict(models[1].named_parameters())
            for name,parameter in models[2].named_parameters():
                if 'adapter' in name:
                    if '.up.weight' in name:torch.testing.assert_close(parameter.grad,coefficient*static[name].grad,rtol=1e-11,atol=1e-13)
                    else:self.assertEqual(float(parameter.grad.norm()),0.)


if __name__=='__main__':unittest.main()
