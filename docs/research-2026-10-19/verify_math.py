"""Numerical algebra checks; no training and no empirical performance claims."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthetic.models import OriginalODELayer
from scripts.diagnose_synthetic import polynomial, layer_diagnostics, time_diagnostics
from synthetic.models import VelocityNet

torch.manual_seed(0)
torch.set_num_threads(1)
errors=[]
for k in (1,2,3):
    layer=OriginalODELayer(6,8,orders=k).double()
    x=torch.randn(6,dtype=torch.float64,requires_grad=True)
    p=layer.operator();s=layer.step_size(x).squeeze()
    ds=torch.autograd.grad(s,x,retain_graph=True)[0]
    a=polynomial(p,s.reshape(1),k)[0]
    b=p@polynomial(p,s.reshape(1),k-1)[0]@x
    formula=a+b[:,None]*ds[None,:]
    actual=torch.autograd.functional.jacobian(layer,x)
    torch.testing.assert_close(actual,formula,atol=1e-12,rtol=1e-12)
    errors.append((actual-formula).abs().max().item())
    def exact(z):
        return torch.matrix_exp(layer.step_size(z).squeeze()*p)@z
    j_exact=torch.autograd.functional.jacobian(exact,x)
    det_formula=torch.exp(s*torch.trace(p))*(1+ds@(p@x))
    torch.testing.assert_close(torch.linalg.det(j_exact),det_formula,atol=1e-11,rtol=1e-11)
omega=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
eye=torch.eye(2,dtype=torch.float64)
rotation={'exact_exp_norm':torch.linalg.matrix_norm(torch.matrix_exp(omega),2).item(),
          'euler_norm':torch.linalg.matrix_norm(eye+omega,2).item(),
          'taylor2_norm':torch.linalg.matrix_norm(eye+omega+omega@omega/2,2).item()}
assert rotation['euler_norm']>1.4 and rotation['taylor2_norm']>1.11
# Unequal Gaussian target variances require expansion as well as contraction.
variance=torch.tensor([.09,4.],dtype=torch.float64)
t=.5
jdiag=(t*variance-(1-t))/((1-t)**2+t*t*variance)
assert jdiag[0]<0 and jdiag[1]>0
# Exercise the posthoc probe code through real modules (nonzero output head).
for kind in ('swiglu','ode_original','ode_spectral_cap'):
    net=VelocityNet(2,kind,width=8,depth=1,gate_hidden=8,ffn_hidden=16)
    with torch.no_grad():net.output.weight.normal_()
    xx=torch.randn(100,2);tt=torch.linspace(.001,.999,100)[:,None]
    probes=layer_diagnostics(net,xx,tt)
    assert len(probes)==(0 if kind=='swiglu' else 2)
    bins=time_diagnostics(net,{'x':xx,'t':tt,'y':torch.randn_like(xx),'oracle':None})
    assert sum(b['count'] for b in bins)==100
result={'jacobian_max_abs_errors':errors,'exact_determinant_formula':'passed',
        'rotation_counterexample':rotation,'gaussian_true_jacobian_at_half_time':jdiag.tolist(),
        'diagnostic_smoke':'SwiGLU, ODE original, ODE cap passed'}
print(json.dumps(result,indent=2))
Path(__file__).with_name('math-verification.json').write_text(json.dumps(result,indent=2)+'\n')
