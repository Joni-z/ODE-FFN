"""Every stored response compression and witness checked by forward differences."""
import time
import numpy as np
import torch
import geometry_study as g
import geometry_math as gm

def audit(root):
    old=g.setup();g.verify(root);formal=g.load(root/'output/result.json');assert formal['status']=='COMPLETE'
    assert len(formal['rows'])==g.load(root/'settings.json')['expected_rows']
    assert len({(v['group'],v['arm']) for v in formal['rows']})==len(formal['rows'])
    out=root/'audit';out.mkdir();tick=time.monotonic();space=g.frames(root);models={a:gm.literal_model(old.model_for(a)) for a in g.ARMS}
    before={a:old.fingerprint(m) for a,m in models.items()};immutable={str(p):g.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
    checked=0;blocks=0;max_block=0.;max_rayleigh=0.;max_scaled=0.
    with np.load(g.PARENT/'states.npz') as states:
        for group in formal['groups']:
            assert time.monotonic()-tick<1800
            domain,j,image_index=group['domain'],group['index'],group['image'];frame=space[group['frame']]
            image=torch.from_numpy(states[f'{domain}_{j}'][image_index:image_index+1]).double();t=j/128
            with np.load(root/'output/groups'/f'{group["name"]}.npz') as f:data={k:f[k] for k in f.files}
            basis=torch.from_numpy(data['basis']);assert basis.shape==(frame.dimension,24)
            np.testing.assert_allclose(data['basis'].T@data['basis'],np.eye(24),rtol=0,atol=1e-11)
            for arm in group['arms']:
                row=next(v for v in formal['rows'] if v['group']==group['name'] and v['arm']==arm)
                m=data[arm+'__compressed_transpose'];eig=np.linalg.eigvalsh((m+m.T)/2);w=data[arm+'__witness'];coords=data['basis'].T@w
                np.testing.assert_allclose(eig,data[arm+'__eigenvalues'],rtol=1e-11,atol=1e-11)
                np.testing.assert_allclose(data['basis']@coords,w,rtol=0,atol=1e-11);assert abs(float(w@w)-1)<1e-11
                expected=dict(min_ritz=float(eig[0]),negative_symmetric_energy=float(np.minimum(eig,0).dot(np.minimum(eig,0))),
                    skew_fraction=float(np.linalg.norm((m-m.T)/2))/(float(np.linalg.norm(m))+1e-30),compressed_operator_norm=float(np.linalg.svd(m,compute_uv=False)[0]))
                for k,value in expected.items():np.testing.assert_allclose(row[k],value,rtol=1e-11,atol=1e-11)
                np.testing.assert_allclose(row['rayleigh'],coords@m@coords,rtol=1e-10,atol=1e-10)
                assert row['negative_hinge']==max(0.,-row['rayleigh'])
                difference=gm.finite_compression(models[arm],image,t,frame,basis,h=1e-4).numpy()
                np.testing.assert_allclose(difference,m,rtol=3e-5,atol=3e-7)
                max_block=max(max_block,float(abs(difference-m).max()));max_scaled=max(max_scaled,float(np.max(abs(difference-m)/(3e-7+3e-5*abs(m)))));blocks+=1
                for h in (1e-4,2e-4):
                    value=gm.finite_rayleigh(models[arm],image,t,frame,torch.from_numpy(w),h)
                    np.testing.assert_allclose(value,row['rayleigh'],rtol=3e-5,atol=3e-7)
                    max_rayleigh=max(max_rayleigh,abs(value-row['rayleigh']));checked+=1
                g.write(out/'progress.json',dict(blocks=blocks,rayleigh_checks=checked,expected_blocks=len(formal['rows']),seconds=time.monotonic()-tick))
    for arm,model in models.items():assert old.fingerprint(model)==before[arm]
    for p,digest in immutable.items():assert g.sha(p)==digest
    assert blocks==len(formal['rows']) and checked==2*blocks;g.verify(root)
    g.write(out/'result.json',dict(status='PASS',finite_difference_blocks=blocks,finite_difference_rayleigh_checks=checked,
        maximum_block_absolute_error=max_block,maximum_block_tolerance_scaled_error=max_scaled,maximum_rayleigh_absolute_error=max_rayleigh,
        formal_files_unchanged=len(immutable),model_tensors_unchanged=True,full_ambient_Jacobian_reconstructed=False,seconds=time.monotonic()-tick))
    print('IMAGE_RESPONSE_AUDIT_PASS',flush=True)
