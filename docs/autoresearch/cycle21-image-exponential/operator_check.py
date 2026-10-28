"""Independent NumPy/SciPy operator audit without importing the image model."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
import study
def bitwise(x,y):assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
def check_operators(root):
    history=study.load(root/'operator-history.json');assert len(history)==48
    for r in history:
        with np.load(root/'operators'/f"{r['family']}-{r['step']:07d}.npz") as bank:
            prefix=study.token(r['layer']);p=bank[prefix+'_P'];a=r['c']*p;e=scipy.linalg.expm(a);t=np.eye(64)+a
            if r['order']==2:t=t+(a@a)/2
            np.testing.assert_allclose(bank[prefix+'_E'],e,rtol=1e-11,atol=1e-11);np.testing.assert_allclose(bank[prefix+'_T'],t,rtol=1e-12,atol=1e-12)
            bitwise(bank[prefix+'_E32'],bank[prefix+'_E'].astype(np.float32));bitwise(bank[prefix+'_T32'],bank[prefix+'_T'].astype(np.float32))
        checkpoint=torch.load(study.paths(root)['image_parent']/'output/runs'/r['family']/f"step_{r['step']:07d}.pt",weights_only=False,map_location='cpu');block,number,part=r['layer'].split('.');actual=checkpoint['model'][f'{block}.{number}.mlp.{part}.proj.weight'].double().numpy();bitwise(p,actual)
        sv=np.linalg.svd(p,compute_uv=False);np.testing.assert_allclose(sv,r['P_singular_values'],rtol=1e-10,atol=1e-12)
        eig=list(np.linalg.eigvals(p));stored=np.asarray(r['P_eigen_real'])+1j*np.asarray(r['P_eigen_imag'])
        for value in stored:
            index=int(np.argmin(np.abs(np.asarray(eig)-value)));assert abs(eig.pop(index)-value)<1e-9
        h=np.eye(64)+a/2
        if r['order']==2:h=h+(a@a)/8
        half=scipy.linalg.expm(a/2);norm=np.linalg.norm(e)
        values=dict(P_sigma_max=sv[0],numerical_abscissa=np.linalg.eigvalsh((p+p.T)/2).max(),spectral_abscissa=max(r['P_eigen_real']),nonnormality=np.linalg.norm(p.T@p-p@p.T)/np.linalg.norm(p)**2,A_norm=np.linalg.norm(a,2),polynomial_norm=np.linalg.norm(t,2),exponential_norm=np.linalg.norm(e,2),relative_Frobenius_error=np.linalg.norm(e-t)/norm,polynomial_semigroup_defect=np.linalg.norm(h@h-t)/norm)
        for key,value in values.items():np.testing.assert_allclose(value,r[key],rtol=1e-9,atol=1e-11)
        assert np.linalg.norm(half@half-e)/norm<1e-11 and r['exponential_semigroup_defect']<1e-11
    return len(history)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();torch.set_num_threads(1);study.verify(a.root)
    count=check_operators(a.root);study.verify(a.root);study.write(a.output,dict(status='PASS',operator_records=count,independent='NumPy/SciPy expm/SVD/eigenvalues and checkpoint tensors',image_model_imported=False))
    print('OPERATOR_CHECK_PASS',count,flush=True)
