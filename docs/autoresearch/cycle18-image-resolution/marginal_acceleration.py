"""Check an exact Gaussian marginal-flow identity; no trained model or sampler."""
import argparse,json
from pathlib import Path
import torch
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();rows=[]
for value in (.2,.5,.8):
    t=torch.tensor(value,dtype=torch.float64,requires_grad=True);y=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    variance=t*t+(1-t)**2;velocity=(2*t-1)*y/variance
    dt,dy=torch.autograd.grad(velocity,(t,y));acceleration=dt+dy*velocity
    expected=y/variance**2
    torch.testing.assert_close(acceleration,expected,atol=1e-12,rtol=1e-12)
    rows.append(dict(t=value,y=.7,conditional_acceleration=0.,marginal_velocity=float(velocity.detach()),
        material_acceleration=float(acceleration.detach()),exact=float(expected.detach())))
assert rows[1]['marginal_velocity']==0 and abs(rows[1]['material_acceleration']-2.8)<1e-12
assert not args.output.exists();result=dict(status='PASS',scope='Closed Gaussian identity only; not a trained experiment',rows=rows)
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
