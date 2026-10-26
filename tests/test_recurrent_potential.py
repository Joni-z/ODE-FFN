"""Matched functions and tied-gradient identities for recurrent potential models."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from synthetic.recurrent_potential import build
from synthetic.conservative_readout import compute_budget
from scripts.run_recurrent_depth import visit_diagnostics


class RecurrentPotentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_matched_nonzero_functions_and_tied_gradients(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            for readout in ('direct','gradient'):
                with self.subTest(family=family,readout=readout):
                    shared=build(3,family,'shared_'+readout).double()
                    cloned=build(3,family,'cloned_'+readout).double()
                    with torch.no_grad():
                        values=torch.linspace(-.11,.13,192).reshape(3,64)
                        for model in (shared,cloned):model.raw.raw.output.weight.copy_(values)
                    x=torch.tensor([[-.3,.7,.1],[.4,-.6,.8]],dtype=torch.float64,requires_grad=True)
                    t=torch.tensor([[.2],[.6]],dtype=torch.float64)
                    a,b=shared(x,t),cloned(x,t)
                    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
                    torch.testing.assert_close(torch.autograd.grad(a.sum(),x,retain_graph=True)[0],
                                               torch.autograd.grad(b.sum(),x,retain_graph=True)[0],rtol=1e-10,atol=1e-11)
                    ga=dict(zip(dict(shared.named_parameters()),torch.autograd.grad(a.square().mean(),tuple(shared.parameters()))))
                    gb=dict(zip(dict(cloned.named_parameters()),torch.autograd.grad(b.square().mean(),tuple(cloned.parameters()))))
                    for name,value in ga.items():
                        if '.blocks.0.' in name:
                            expected=sum(gb[name.replace('.blocks.0.',f'.blocks.{i}.')] for i in range(3))
                        else:expected=gb[name]
                        torch.testing.assert_close(value,expected,rtol=1e-8,atol=1e-10)

    def test_initial_state_rng_and_unique_parameters(self):
        for family in ('ode_k2','swiglu128','swiglu192'):
            a=build(16,family,'shared_gradient');rng_a=torch.get_rng_state().clone()
            b=build(16,family,'cloned_gradient');rng_b=torch.get_rng_state().clone()
            self.assertTrue(torch.equal(rng_a,rng_b))
            self.assertEqual(len(a.raw.raw.blocks),1);self.assertEqual(len(b.raw.raw.blocks),3)
            aa=a.raw.raw.blocks[0].state_dict()
            for block in b.raw.raw.blocks:
                for name,value in block.state_dict().items():self.assertTrue(torch.equal(value,aa[name]))
            self.assertEqual(len({id(p) for p in a.parameters()}),len(list(a.parameters())))
            ca,cb=compute_budget(a),compute_budget(b)
            self.assertLess(ca['parameters'],cb['parameters'])
            self.assertEqual(ca['dense_forward_macs'],cb['dense_forward_macs'])

    def test_actual_six_visits_for_two_or_six_unique_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            np.savez(tmp/'probe.npz',x=np.linspace(-1,1,64,dtype=np.float32).reshape(32,2),
                     t=np.full((32,1),.4,dtype=np.float32))
            for layout,unique in (('shared',2),('cloned',6)):
                model=build(2,'ode_k2',layout+'_gradient')
                self.assertEqual(len(list(model.ode_layers())),unique)
                rows=visit_diagnostics(model,tmp)
                self.assertEqual(len(rows),6)
                if layout=='shared':
                    self.assertEqual(sorted(r['visit'] for r in rows),[1,1,2,2,3,3])
                else:self.assertTrue(all(r['visit']==1 for r in rows))

    def test_native_step_boundary(self):
        m=build(2,'ode_k2','shared_gradient')
        with self.assertRaises(ValueError):m(torch.zeros(2,2),torch.full((2,1),.4),updates=6)


if __name__=='__main__':unittest.main()
