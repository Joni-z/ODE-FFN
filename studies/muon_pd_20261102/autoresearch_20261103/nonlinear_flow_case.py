"""Known scalar-flow consistency witness, not learning or quality evidence."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--save',required=True);a=p.parse_args()
h=.7;rate=-.4;rows=[]
for tau in [.2,.1,.05,.025,.0125]:
    exact=h/(1-rate*tau*h)
    frozen_exact=h*math.exp(tau*rate*h)
    frozen=h+tau*rate*h*h+.5*tau*tau*rate*rate*h**3
    feedback=h+tau*rate*h*h+tau*tau*rate*rate*h**3
    rows.append(dict(tau=tau,exact_nonlinear=exact,frozen_exact=frozen_exact,
                     frozen_taylor_error=abs(frozen-frozen_exact),
                     frozen_nonlinear_error=abs(frozen-exact),
                     feedback_nonlinear_error=abs(feedback-exact)))
for key,order in [('frozen_taylor_error',3),('frozen_nonlinear_error',2),('feedback_nonlinear_error',3)]:
    ratio=rows[-2][key]/rows[-1][key]
    assert abs(ratio-2**order)<.12,(key,ratio)
out=Path(a.save);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(h=h,p=rate,rows=rows,interpretation='Scalar known-flow consistency, no trained model or quality claim.'),indent=2)+'\n')
print(out)
