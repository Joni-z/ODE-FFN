"""Independent local flow, derivative and cost witnesses; no quality training."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from invariant_clocks import InvariantClockLayer,InvariantClockFFN,IdentitySiLU


def verify():
    torch.set_num_threads(1);tick=time.monotonic();errors={};cases=0;integrations=[]
    def compare(name,a,b,atol=2e-10):
        a=torch.as_tensor(a,dtype=torch.float64).detach();b=torch.as_tensor(b,dtype=torch.float64).detach().to(a)
        error=float((a-b).abs().max());errors[name]=max(errors.get(name,0.),error)
        torch.testing.assert_close(a,b,rtol=2e-10,atol=atol);return error
    def literal(layer,x,duration):
        angle=duration*layer.rates(x);d=len(x);index=torch.arange(d//2);matrix=x.new_zeros(d,d)
        if layer.kind=='normalized_euler':angle=torch.atan(angle)
        matrix[2*index,2*index+1]=-angle;matrix[2*index+1,2*index]=angle
        return (torch.eye(d,dtype=x.dtype)+matrix)@x if layer.kind=='euler' else torch.matrix_exp(matrix)@x
    def uv(layer,x):
        d=len(x);m=d//2;u=x.new_zeros(d,m);v=x.new_zeros(m,d)
        for j in range(m):u[2*j:2*j+2,j]=torch.stack((-x[2*j+1],x[2*j]));v[j,2*j:2*j+2]=x[2*j:2*j+2]
        return u,v
    for d in (2,4,6):
        torch.manual_seed(0);reference=InvariantClockLayer(d,2*d).double()
        with torch.no_grad():reference.frequency.copy_(torch.linspace(-.7,.5,d//2))
        points=[torch.zeros(d,dtype=torch.float64),torch.linspace(-.8,1.1,d,dtype=torch.float64),torch.cos(torch.arange(d,dtype=torch.float64))]
        for kind in ('rotation','euler','normalized_euler'):
            layer=InvariantClockLayer(d,2*d,kind).double();layer.load_state_dict(reference.state_dict())
            for original in points:
                x=original.clone().requires_grad_(True);output=layer(x,.7);expected=literal(layer,x,.7)
                compare('literal_output',output,expected)
                actual_j=torch.autograd.functional.jacobian(lambda value:layer(value,.7),x,vectorize=True)
                compare('complete_input_Jacobian',actual_j,layer.complete_jacobian(x,.7))
                probe=torch.linspace(.4,1.2,d,dtype=torch.float64);parameters=(x,*layer.parameters())
                a=torch.autograd.grad((output*probe).sum(),parameters);b=torch.autograd.grad((expected*probe).sum(),parameters)
                for left,right in zip(a,b):compare('literal_input_and_parameter_gradients',left,right)
                if kind=='rotation':
                    phi=layer(x,.7);compare('pair_invariants',layer.invariants(phi),layer.invariants(x))
                    compare('conserved_clocks',layer.clocks(layer.invariants(phi)),layer.clocks(layer.invariants(x)))
                    compare('inverse',layer(phi,-.7),x);compare('semigroup',layer(layer(x,.3),.2),layer(x,.5))
                    clock=torch.tensor(.7,dtype=torch.float64)
                    derivative=torch.autograd.functional.jacobian(lambda value:layer(x,value),clock)
                    compare('autonomous_field',derivative,layer.vector_field(phi))
                    u,v=uv(layer,x);feedback=u@layer.rate_derivative(x)@v
                    compare('VU_zero',v@u,torch.zeros(d//2,d//2,dtype=x.dtype))
                    compare('feedback_square_zero',feedback@feedback,torch.zeros(d,d,dtype=x.dtype))
                    compare('determinant',torch.linalg.det(actual_j),torch.tensor(1.,dtype=x.dtype))
                    sigma=torch.linalg.svdvals(.7*feedback)[0];gain=(torch.sqrt(sigma*sigma+4)+sigma)/2
                    compare('full_gain_formula',torch.linalg.svdvals(actual_j)[0],gain)
                elif kind=='normalized_euler':
                    compare('normalized_invariants',layer.invariants(output),layer.invariants(x))
                    compare('normalized_determinant',torch.linalg.det(actual_j),torch.tensor(1.,dtype=x.dtype))
                cases+=1
        # Independently integrate the actual re-evaluated NumPy vector field.
        layer=reference;weights=[p.detach().numpy().copy() for p in layer.clock.parameters()];omega=layer.frequency.detach().numpy().copy()
        def field(_,y):
            pairs=y.reshape(-1,2);actions=.5*np.sum(pairs*pairs,axis=1)
            w1,b1,w2,b2=weights;z=w1@actions+b1;rates=omega*(.2+.8*expit((w2@(z*expit(z))+b2)/10))
            return (rates[:,None]*np.stack((-pairs[:,1],pairs[:,0]),axis=1)).ravel()
        initial=points[1].numpy();solution=solve_ivp(field,(0,.7),initial,method='DOP853',rtol=1e-11,atol=1e-12)
        assert solution.success
        error=compare('independent_integration',solution.y[:,-1],layer(points[1],.7),atol=2e-9)
        integrations.append(dict(width=d,nfev=solution.nfev,maximum_error=error))
    # A bounded sigmoid clock exactly realizable through SiLU(z)-SiLU(-z).
    witness=InvariantClockLayer(2,2).double()
    with torch.no_grad():
        witness.frequency.fill_(1);witness.clock[0].weight.copy_(torch.tensor([[1.],[-1.]],dtype=torch.float64))
        witness.clock[0].bias.copy_(torch.tensor([-.5,.5],dtype=torch.float64));witness.clock[2].weight.copy_(torch.tensor([[50.,-50.]],dtype=torch.float64));witness.clock[2].bias.zero_()
    x=torch.tensor([1.,0.],dtype=torch.float64);jac=witness.complete_jacobian(x);gain=float(torch.linalg.svdvals(jac)[0])
    compare('noncontractive_gain',gain,(1+math.sqrt(5))/2);compare('witness_norm',witness(x).norm(),x.norm())
    # Cross-mode rates need not be a gradient: volume does not imply symplectic.
    cross=InvariantClockLayer(4,2).double()
    with torch.no_grad():
        cross.frequency.fill_(1);cross.clock[0].weight.copy_(torch.tensor([[0.,1.],[0.,-1.]],dtype=torch.float64))
        cross.clock[0].bias.copy_(torch.tensor([-.5,.5],dtype=torch.float64));cross.clock[2].weight.copy_(torch.tensor([[50.,-50.],[0.,0.]],dtype=torch.float64));cross.clock[2].bias.zero_()
    point=torch.tensor([1.,0.,1.,0.],dtype=torch.float64);dj=cross.complete_jacobian(point)
    form=torch.block_diag(torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64),torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64))
    symplectic_defect=float(torch.linalg.matrix_norm(dj.T@form@dj-form));assert symplectic_defect>.1
    equal_norm=[torch.tensor(v,dtype=torch.float64) for v in ([math.sqrt(2),0,0,0],[0,0,math.sqrt(2),0])]
    compare('equal_global_norm',equal_norm[0].norm(),equal_norm[1].norm())
    pair_clock_difference=float((cross.rates(equal_norm[0])-cross.rates(equal_norm[1])).abs().max());assert pair_clock_difference>.1
    negative={}
    for kind in ('euler','normalized_euler'):
        control=InvariantClockLayer(2,2,kind).double();control.load_state_dict(witness.state_dict())
        negative[kind]=float((control(control(x,.3),.4)-control(x,.7)).norm());assert negative[kind]>1e-4
    # Full two-solve FFN and every parameter derivative against literal solves.
    torch.manual_seed(0);ffn=InvariantClockFFN(4,8).double()
    with torch.no_grad():
        ffn.ode1.frequency.copy_(torch.tensor([.4,-.3],dtype=torch.float64));ffn.ode2.frequency.copy_(torch.tensor([-.2,.5],dtype=torch.float64))
    value=torch.linspace(-1,1,4,dtype=torch.float64).requires_grad_(True)
    result=ffn(value);explicit=literal(ffn.ode2,torch.nn.functional.silu(literal(ffn.ode1,value,1.)),1.)
    compare('full_FFN_literal',result,explicit)
    direction=torch.sin(torch.arange(4,dtype=torch.float64)+.4)
    left=torch.autograd.grad((result*direction).sum(),(value,*ffn.parameters()));right=torch.autograd.grad((explicit*direction).sum(),(value,*ffn.parameters()))
    for a,b in zip(left,right):compare('full_FFN_gradients',a,b)
    assert all(float(v.norm())>1e-12 for v in left[1:]),'nonzero-frequency active routes'
    # Actual width64, zero-frequency initial pairing and executed dense work.
    torch.manual_seed(0);sample=torch.randn(3,64,requires_grad=True);wide=InvariantClockFFN()
    for kind in ('rotation','euler','normalized_euler'):
        body=InvariantClockFFN(kind=kind);body.load_state_dict(wide.state_dict())
        torch.testing.assert_close(body(sample),torch.nn.functional.silu(sample),rtol=0,atol=0)
    for kind in ('silu_identity','fixed06_identity'):
        torch.testing.assert_close(IdentitySiLU(kind=kind)(sample),torch.nn.functional.silu(sample),rtol=0,atol=0)
    class Count(TorchDispatchMode):
        def __init__(self):self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if func==torch.ops.aten.mm.default:a,b=args[:2];self.macs+=a.shape[0]*a.shape[1]*b.shape[1]
            elif func==torch.ops.aten.addmm.default:a,b=args[1:3];self.macs+=a.shape[0]*a.shape[1]*b.shape[1]
            return func(*args,**(kwargs or {}))
    with Count() as forward:out=wide(sample)
    with Count() as backward:torch.autograd.grad(out.square().sum(),(sample,*wide.parameters()))
    cost=dict(parameters=sum(v.numel() for v in wide.parameters()),forward_matrix_MACs=forward.macs//3,forward_backward_matrix_MACs=(forward.macs+backward.macs)//3)
    assert cost==dict(parameters=16768,forward_matrix_MACs=16384,forward_backward_matrix_MACs=49152),cost
    return dict(status='PASS',literal_derivative_cases=cases,independent_integrations=integrations,maximum_errors=errors,
        bounded_clock_noncontractive_gain=gain,symplectic_defect_witness=symplectic_defect,equal_norm_clock_difference=pair_clock_difference,
        controls_original_semigroup_defect=negative,full_FFN_active_routes=True,actual_width64_initial_pairing=True,cost=cost,seconds=time.monotonic()-tick,
        training_or_quality_evaluated=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
    answer=verify();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n');print(json.dumps(answer),flush=True)
