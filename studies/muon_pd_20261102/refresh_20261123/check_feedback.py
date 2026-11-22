"""Test native function, native gradients and the two hidden drift evaluations."""
from pathlib import Path
import sys
from odd_trial import base, build_model, torch


def check(dim):
    torch.manual_seed(0)
    native=build_model(dim,'odd_swiglu192')
    torch.manual_seed(0)
    proposed=build_model(dim,'odd_feedback192')
    with torch.no_grad():
        native.output.weight.normal_(std=.1)
        proposed.output.weight.copy_(native.output.weight)
    x=torch.randn(128,dim);t=torch.rand(128,1)
    def forward_and_count(model):
        calls=[]
        hooks=[layer.register_forward_hook(lambda module,args,out: calls.append(module))
               for b in model.blocks for layer in (b.ffn.w12,b.ffn.w3)]
        try:
            value=model(x,t)
        finally:
            for handle in hooks:
                handle.remove()
        assert len(calls)==len(set(calls))==2*len(model.blocks)
        return value,len(calls)
    y,native_calls=forward_and_count(native)
    z,feedback_calls=forward_and_count(proposed)
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    y.square().mean().backward();z.square().mean().backward()
    own=dict(proposed.named_parameters())
    for name,p in native.named_parameters():
        torch.testing.assert_close(p.grad,own[name].grad,atol=1e-7,rtol=1e-6)
    norms=[]
    for b in proposed.blocks:
        f=b.ffn
        norms.append(f.feedback.grad.norm().item())
        assert norms[-1]>0
        source=torch.randn(64,f.w12.in_features)
        with torch.no_grad():
            f.feedback.uniform_(-1,1)
        g,v=f.w12(source).chunk(2,-1)
        def drift(s):
            return torch.nn.functional.silu(g+f.feedback*s/torch.sqrt(1+s.square()))*v
        state=torch.zeros_like(v)
        midpoint=state+.5*drift(state)
        reference=f.w3(state+drift(midpoint))
        torch.testing.assert_close(reference,f(source),rtol=1e-5,atol=1e-6)
    prediction=proposed(x,t)
    assert (prediction-z).abs().max()>0
    torch.testing.assert_close(prediction,-proposed(-x,t),atol=0,rtol=0)
    return dict(dim=dim,zero_feedback_exact_native=True,all_shared_gradients_match=True,
                native_ffn_linear_calls=native_calls,feedback_ffn_linear_calls=feedback_calls,feedback_parameters=sum(b.ffn.feedback.numel() for b in proposed.blocks),
                feedback_gradient_norms=norms,nonzero_gain_matches_two_stage_midpoint=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    base.dump(Path(sys.argv[1]),[check(d) for d in (2,16)])
