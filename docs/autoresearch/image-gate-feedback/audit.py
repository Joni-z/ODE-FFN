"""Independent all-probe AD and manual native-state replay; no analytic Jacobian."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
import study

def silu(x):return x*torch.sigmoid(x)
def affine(x,m):return x@m.weight.T+(0 if m.bias is None else m.bias)
def clock(layer,x):return (layer.shift+layer.scale*torch.sigmoid(affine(silu(affine(x,layer.t_func[0])),layer.t_func[2])/layer.tau)).squeeze(-1)
def layer(m,x,held=None):
    s=clock(m,x) if held is None else held
    answer=x;term=x
    for j in range(m.orders):term=affine(term,m.proj)*s[...,None]/(j+1);answer=answer+term
    return answer
def core(m,x,held=None):
    if hasattr(m,'ode1'):return layer(m.ode2,silu(layer(m.ode1,x,None if held is None else held[...,0])),None if held is None else held[...,1])
    if hasattr(m,'w12'):
        a,b=affine(x,m.w12).chunk(2,dim=-1);return affine(silu(a)*b,m.w3)
    return affine(silu(affine(x,m.fc1)),m.fc2)
def held_values(m,x):
    s1=clock(m.ode1,x);z=silu(layer(m.ode1,x,s1));return torch.stack((s1,clock(m.ode2,z)),dim=-1).detach()
def main(root):
    torch.set_num_threads(1);study.verify(root);assert study.load(root/'output/status.json')['state']=='COMPLETE'
    out=root/'audit';out.mkdir();tick=time.monotonic();rows=[]
    for cell in study.CELLS:
        model,captures,inputs,generated=study.capture(cell,manual=True);folder=root/'output'/cell
        np.testing.assert_array_equal(generated,np.load(folder/'generated-prefix.npy'))
        with np.load(folder/'inputs.npz') as f:
            assert set(f.files)==set(inputs)
            for k,v in inputs.items():np.testing.assert_array_equal(v,f[k])
        for key,r in sorted(captures.items()):
            assert time.monotonic()-tick<2000
            with np.load(folder/f'{key}-capture.npz') as f:
                assert set(r)==set(f.files)
                for k,v in r.items():np.testing.assert_array_equal(v,f[k])
            m=copy.deepcopy(study.module_for(model,key)).double().eval().requires_grad_(False)
            x=torch.from_numpy(r['h']).double();local='-ode' in key;fn=layer if local else core
            jac=torch.func.vmap(torch.func.jacrev(lambda h:fn(m,h)))(x)
            expected=dict(sv_full=torch.linalg.svdvals(jac).numpy())
            if local or hasattr(m,'ode1'):
                held=clock(m,x).detach() if local else held_values(m,x)
                ref=torch.func.vmap(torch.func.jacrev(lambda h,c:fn(m,h,c),argnums=0))(x,held)
                expected['sv_ref']=torch.linalg.svdvals(ref).numpy()
                if local:
                    gradient=torch.func.vmap(torch.func.jacrev(lambda h:clock(m,h)))(x)
                    difference=jac-ref
                    expected.update(s=held.numpy(),gradient_norm=gradient.norm(dim=-1).numpy(),
                        feedback_norm=torch.linalg.matrix_norm(difference,ord=2).numpy(),P_sigma_max=np.array(float(torch.linalg.svdvals(m.proj.weight)[0])))
                    valid=expected['sv_ref'][:,-1]>np.finfo(np.float64).eps*64*expected['sv_ref'][:,0]
                    chi=np.full(256,np.nan)
                    if valid.any():
                        # trace(R^-1 (J-R)) is the determinant multiplier minus1.
                        chi[valid]=torch.linalg.solve(ref[valid],difference[valid]).diagonal(dim1=-2,dim2=-1).sum(-1).numpy()
                    expected.update(base_inverse_valid=valid,chi=chi)
                else:expected['sv_feedback']=torch.linalg.svdvals(jac-ref).numpy()
            with np.load(folder/f'{key}-spectra.npz') as f:
                assert set(f.files)==set(expected)
                maximum=0.
                for k,v in expected.items():
                    if v.dtype==bool:np.testing.assert_array_equal(v,f[k])
                    else:
                        np.testing.assert_allclose(v,f[k],rtol=1e-8,atol=1e-10,equal_nan=True)
                        finite=np.isfinite(v)
                        if finite.any():maximum=max(maximum,float(np.max(np.abs(v[finite]-f[k][finite]))))
            native=fn(m,x).detach().numpy();direct=m(x);direct=direct[0] if isinstance(direct,tuple) else direct
            np.testing.assert_allclose(native,direct.detach().numpy(),rtol=1e-10,atol=1e-10)
            reported=next(v for v in study.load(folder/'result.json')['rows'] if v['group']==key)
            error=native-r['native_y'].astype(np.float64)
            np.testing.assert_allclose([reported['native_to_promoted_max'],reported['native_to_promoted_RMS']],
                [np.max(np.abs(error)),np.sqrt(np.mean(error**2))],rtol=1e-8,atol=1e-10)
            rows.append(dict(cell=cell,group=key,probes=256,captures_bitwise=True,AD_spectra_max_error=maximum))
        print(study.load(folder/'result.json')['cell'],'all captures and AD checked',flush=True)
        study.write(out/'status.json',dict(state='RUNNING',groups=len(rows),seconds=time.monotonic()-tick))
    assert len(rows)==480 and sum(r['probes'] for r in rows)==122880
    study.verify(root);study.write(out/'result.json',dict(status='PASS',rows=rows,groups=480,probes=122880,
        native_endpoints_bitwise=8,unchanged_parameters_buffers=True,seconds=time.monotonic()-tick))
    study.write(out/'status.json',dict(state='COMPLETE',groups=len(rows),seconds=time.monotonic()-tick))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
