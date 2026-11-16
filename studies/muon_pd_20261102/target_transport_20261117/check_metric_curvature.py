"""Analytic Gaussian FM counterexample and exact metric-error accounting."""
import argparse
import math
import numpy as np
from common import base,torch,dump
from learned_transport import SensitivityField,BlendedField,integrate,squared_sw2


class GaussianField(torch.nn.Module):
    def __init__(self,odd: bool):
        super().__init__();self.odd=odd

    def forward(self,x,t):
        variance=(1-t).square()+t.square()
        shift=.2*(2*t-1) if self.odd else .1
        return (2*t-1)/variance*x+shift


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    source=torch.linspace(-2,2,64,dtype=torch.float64)[:,None]
    native,crossed=GaussianField(False),GaussianField(True)
    mean=.1*math.sqrt(2)*math.asinh(1)
    a=base.train.generated_samples(native,source,128)
    b=base.train.generated_samples(crossed,source,128)
    sensitivity=SensitivityField(native,crossed)
    augmented,_=integrate(base.train.generated_samples,sensitivity,
                           torch.cat((source,torch.zeros_like(source)),1))
    endpoint_error=max(float((a-source-mean).abs().max()),float((b-source).abs().max()))
    tangent_error=float((augmented[:,1:]+mean).abs().max())
    assert torch.equal(a,augmented[:,:1]) and endpoint_error<1e-4 and tangent_error<1e-4
    target=source.numpy();directions=np.ones((1,1));rows=[]
    l0=squared_sw2(a.numpy(),target,directions)
    derivative=2*float(np.mean((a.numpy()-target)*augmented[:,1:].numpy()))
    for coefficient in (.125,.25,.5,1.):
        actual=base.train.generated_samples(BlendedField(native,crossed,coefficient),source,128).numpy()
        linear_endpoint=a.numpy()+coefficient*augmented[:,1:].numpy()
        la=squared_sw2(actual,target,directions)
        lh=squared_sw2(linear_endpoint,target,directions)
        transport=la-lh
        curvature=lh-l0-coefficient*derivative
        closure=(la-l0-coefficient*derivative)-(transport+curvature)
        assert abs(closure)<1e-14 and abs(transport)<1e-12
        rows.append(dict(coefficient=coefficient,actual_loss=la,linear_endpoint_loss=lh,
            linear_loss_change=coefficient*derivative,transport_remainder=transport,
            functional_remainder=curvature,closure=closure))
    result=dict(passed=True,field_mse_A=.01,field_mse_B=.04/3,
        population_w2_A=mean,population_w2_B=0.,native_heun_endpoint_max_error=endpoint_error,
        native_tangent_max_error=tangent_error,all_native_primal_exact=True,rows=rows,
        original_primary_gate_unchanged=True,learned_checkpoints_used=0)
    from pathlib import Path
    assert not Path(args.output).exists()
    dump(args.output,result)
    print('GAUSSIAN_FIELD_METRIC_CONTROL_PASS',endpoint_error,tangent_error)


if __name__=='__main__':main()
