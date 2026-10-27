"""Post-result coordinate algebra only; no model, data or optimizer change."""
import json

import torch

torch.set_default_dtype(torch.float64)
p=torch.linspace(-.3,.4,9).reshape(3,3).requires_grad_(True)
c=torch.linspace(.1,-.2,9).reshape(3,3).requires_grad_(True)
gp=torch.linspace(-.2,.3,9).reshape(3,3)
gb=torch.linspace(.4,-.3,9).reshape(3,3)
loss=(p*gp).sum()+((p@p+c)*gb).sum()
gradient_p,gradient_c=torch.autograd.grad(loss,(p,c))
t=lambda h:p.detach()@h+h@p.detach()
adjoint=lambda g:p.detach().T@g+g@p.detach().T
torch.testing.assert_close(gradient_p,gp+adjoint(gb),rtol=0,atol=1e-14)
torch.testing.assert_close(gradient_c,gb,rtol=0,atol=1e-14)
theta=torch.cat((p.detach().reshape(-1),c.detach().reshape(-1)))
def coordinates(value):
    pp=value[:9].reshape(3,3);cc=value[9:].reshape(3,3)
    return torch.cat((pp.reshape(-1),(pp@pp+cc).reshape(-1)))
j=torch.autograd.functional.jacobian(coordinates,theta)
metric=j@j.T;gradient=torch.cat((gp.reshape(-1),gb.reshape(-1)))
expected_p=-gp-adjoint(gb);expected_b=-t(gp)-gb-t(adjoint(gb))
expected=torch.cat((expected_p.reshape(-1),expected_b.reshape(-1)))
torch.testing.assert_close(-metric@gradient,expected,atol=1e-14,rtol=0)
rate=gradient@expected;identity=-torch.sum((gp+adjoint(gb))**2)-torch.sum(gb**2)
torch.testing.assert_close(rate,identity,atol=1e-14,rtol=0)
torch.testing.assert_close(torch.linalg.det(metric),torch.tensor(1.),atol=1e-13,rtol=0)
assert torch.linalg.eigvalsh(metric).min()>0
print(json.dumps(dict(status='PASS',post_result_explanation=True,no_fitted_data=True,
    pullback_gradient_max=float((gradient_p-gp-adjoint(gb)).abs().max()),
    metric_pushforward_max=float((-metric@gradient-expected).abs().max()),
    loss_rate_discrepancy=float((rate-identity).abs()),determinant=float(torch.linalg.det(metric)),
    minimum_eigenvalue=float(torch.linalg.eigvalsh(metric).min()))))
