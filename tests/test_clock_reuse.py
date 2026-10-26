"""Clock reuse semantics, full differential, budget and checkpoint behavior."""
import io
import unittest
import torch
from synthetic.models import VectorODEFFN,VelocityNet
from synthetic.diagnostics import collect
from synthetic.train import capture_state,restore_model
from scripts.run_architecture_study import macs


class ClockReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_single_clock_evaluation_and_shared_jacobian(self) -> None:
        block=VectorODEFFN(4,3,reference_gate_hidden=4,orders=1,clock_mode='reuse').double()
        calls=[]
        handle=block.ode1.t_func.register_forward_hook(lambda *args:calls.append(1))
        x=torch.randn(4,dtype=torch.float64,requires_grad=True)
        block(x[None]);handle.remove()
        self.assertEqual(len(calls),1)
        gate=lambda v:block.ode1.step_size(v[None])[0]
        s=gate(x);js=torch.autograd.functional.jacobian(gate,x)
        d=torch.diag(s);p1=block.ode1.proj.weight;p2=block.ode2.proj.weight
        eye=torch.eye(4,dtype=torch.float64)
        z=x+d@p1@x;u=torch.nn.functional.silu(z)
        logistic=torch.sigmoid(z)
        act=torch.diag(logistic*(1+z*(1-logistic)))
        expected=(eye+d@p2)@act@(eye+d@p1)+((eye+d@p2)@act@torch.diag(p1@x)+torch.diag(p2@u))@js
        actual=torch.autograd.functional.jacobian(lambda v:block(v[None])[0],x)
        torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-12)

    def test_pairing_and_true_budget_match(self) -> None:
        models={};states={}
        for mode in ('independent','tied','reuse'):
            torch.manual_seed(0)
            models[mode]=VelocityNet(2,'ode_vector',gate_hidden=32,orders=1,ode_clock_mode=mode)
            states[mode]=torch.get_rng_state()
        reference=models['independent'].state_dict()
        for mode in ('tied','reuse'):
            torch.testing.assert_close(states['independent'],states[mode],atol=0,rtol=0)
            for name,value in models[mode].state_dict().items():
                if '.t_func.' not in name or '.ode1.t_func.' in name:
                    torch.testing.assert_close(value,reference[name],atol=0,rtol=0)
        self.assertEqual(models['tied'].parameter_count(),models['reuse'].parameter_count())
        self.assertEqual(models['reuse'].parameter_count(),69282)
        self.assertEqual(macs(models['reuse'],2),67904)
        self.assertEqual(macs(models['tied'],2),macs(models['independent'],2))
        self.assertEqual(macs(models['reuse'],2),macs(VelocityNet(2,'swiglu',ffn_hidden=96),2))
        self.assertIs(models['tied'].blocks[0].ffn.ode1.t_func,models['tied'].blocks[0].ffn.ode2.t_func)
        self.assertFalse(hasattr(models['reuse'].blocks[0].ffn.ode2,'t_func'))

    def test_diagnostics_and_weight_roundtrip_retain_live_gradient(self) -> None:
        for mode in ('tied','reuse'):
            model=VelocityNet(2,'ode_vector',width=8,depth=1,gate_hidden=4,
                gate_reference_hidden=8,orders=1,ode_clock_mode=mode)
            with torch.no_grad():model.output.weight.normal_()
            x=torch.randn(16,2);t=torch.rand(16,1)
            records=collect(model,x,t)
            self.assertEqual(len(records),2)
            if mode=='reuse':self.assertEqual(records[-1]['gate_source'],'provided')
            model(x,t).square().mean().backward()
            self.assertGreater(float(model.blocks[0].ffn.ode1.t_func[0].weight.grad.abs().sum()),0.)
            data=io.BytesIO();torch.save(model.state_dict(),data);data.seek(0)
            clone=VelocityNet(2,'ode_vector',width=8,depth=1,gate_hidden=4,
                gate_reference_hidden=8,orders=1,ode_clock_mode=mode)
            clone.load_state_dict(torch.load(data,weights_only=True))
            torch.testing.assert_close(model(x,t),clone(x,t),atol=0,rtol=0)

    def test_checkpoint_restores_platform_sensitive_nonpersistent_buffers(self) -> None:
        model=VelocityNet(2,'ode_vector',width=8,depth=1,gate_hidden=4,
            gate_reference_hidden=8,orders=1,ode_clock_mode='reuse')
        with torch.no_grad():model.output.weight.normal_()
        optimizer=torch.optim.AdamW(model.parameters())
        state=capture_state(model,optimizer,torch.Generator(),0,'config','data',{})
        clone=VelocityNet(2,'ode_vector',width=8,depth=1,gate_hidden=4,
            gate_reference_hidden=8,orders=1,ode_clock_mode='reuse')
        with torch.no_grad():clone.time.frequencies.mul_(.75)
        restore_model(clone,state)
        torch.testing.assert_close(model.time.frequencies,clone.time.frequencies,atol=0,rtol=0)
        x=torch.randn(16,2);t=torch.rand(16,1)
        torch.testing.assert_close(model(x,t),clone(x,t),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
