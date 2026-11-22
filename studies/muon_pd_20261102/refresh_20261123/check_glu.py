"""Check the candidate against the ordinary residual block on nonzero fields."""
import json
from pathlib import Path
import sys
from odd_trial import base, build_model, torch


def check(dim):
    torch.manual_seed(0)
    ordinary=build_model(dim,'odd_swiglu192')
    torch.manual_seed(0)
    midpoint=build_model(dim,'odd_glu_midpoint192')
    with torch.no_grad():
        ordinary.output.weight.normal_(std=.1)
    midpoint.load_state_dict(ordinary.state_dict(),strict=True)
    x=torch.randn(64,dim)
    t=torch.rand(64,1)
    h=torch.nn.functional.linear(x,midpoint.input.weight).square()+midpoint.input.bias
    temb=midpoint.time(t)
    manual=h
    max_stage=0.
    for eb,mb in zip(ordinary.blocks,midpoint.blocks):
        scale,shift=eb.condition(temb).chunk(2,dim=-1)
        z0=eb.norm(manual)*(1+scale)+shift
        k1=eb.ffn(z0)
        euler=manual+eb.residual_scale*k1
        torch.testing.assert_close(eb(manual,temb),euler,rtol=0,atol=0)
        half=manual+.5*eb.residual_scale*k1
        z1=eb.norm(half)*(1+scale)+shift
        ref=manual+eb.residual_scale*eb.ffn(z1)
        seen=[]
        handle=mb.ffn.register_forward_pre_hook(lambda _,args: seen.append(args[0].detach().clone()))
        actual=mb(manual,temb)
        handle.remove()
        assert len(seen)==2
        torch.testing.assert_close(seen[0],z0,rtol=0,atol=0)
        torch.testing.assert_close(seen[1],z1,rtol=0,atol=0)
        torch.testing.assert_close(actual,ref,rtol=0,atol=0)
        max_stage=max(max_stage,(seen[1]-seen[0]).abs().max().item())
        manual=ref
    y=midpoint(x,t)
    ref=ordinary.output(torch.nn.functional.linear(x,ordinary.input.weight)*ordinary.norm(manual))
    torch.testing.assert_close(y,ref,rtol=0,atol=0)
    torch.testing.assert_close(y,-midpoint(-x,t),rtol=0,atol=0)
    change=(y-ordinary(x,t)).square().mean().sqrt().item()
    assert change>0 and max_stage>0
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in midpoint.parameters())
    return dict(dim=dim,n_parameters=sum(p.numel() for p in midpoint.parameters()),
                exact_manual_midpoint=True,exact_native_euler=True,odd_output=True,
                max_stage_input_change=max_stage,output_rms_change=change,finite_gradients=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    base.dump(Path(sys.argv[1]),[check(dim) for dim in (2,16)])
