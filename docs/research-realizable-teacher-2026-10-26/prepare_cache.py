"""One deterministic input replay and one live-label pass per frozen teacher."""
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from teacher_support import TEACHERS,TIMES,SNAPSHOTS,OLD,BANKS,DATASET,load,write,sha,setup,teacher,signature,targets,draw,predict,replay_hash

def prepare(folder,steps=5000,progress=None):
    setup();folder.mkdir(exist_ok=False);started=time.monotonic()
    from synthetic.population import fingerprint_batch
    from scripts.check_gate_geometry import jacobian
    models={};teacher_paths={};signatures={}
    for name in TEACHERS:
        models[name],cp=teacher(name);teacher_paths[name]=str(cp);signatures[name]=signature(models[name])
    bank,_=targets();rng=torch.Generator().manual_seed(100000);empty=hashlib.sha256(b'').hexdigest();input_digest=empty;label_digests={name:empty for name in TEACHERS}
    replay_input=empty;replay_labels={name:empty for name in TEACHERS}
    cache={key:np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype=np.float32,shape=(steps,512,width)) for key,width in [('x',16),('t',1)]+[('label-'+name,16) for name in TEACHERS]}
    wanted={s for s in SNAPSHOTS if s<=steps}|{0,2,3,steps};snapshots={};source_rng={}
    def record(step):
        snapshots[str(step)]=dict(input_digest=input_digest,label_digests=label_digests.copy(),input_replay_digest=replay_input,label_replay_digests=replay_labels.copy());source_rng[step]=rng.get_state().clone()
    record(0)
    with torch.no_grad():
        for i in range(steps):
            x,t,target,source=draw(bank,rng);cache['x'][i]=x.numpy();cache['t'][i]=t.numpy();input_digest=fingerprint_batch(input_digest,target,source,t)
            replay_input=replay_hash(replay_input,x,t)
            for name,m in models.items():
                y=m(x,t);assert torch.isfinite(y).all();array=y.numpy();cache['label-'+name][i]=array
                h=hashlib.sha256(bytes.fromhex(label_digests[name]));h.update(array.tobytes());label_digests[name]=h.hexdigest()
                replay_labels[name]=replay_hash(replay_labels[name],y)
            if i+1 in wanted:record(i+1)
            if progress and ((i+1)%500==0 or i+1==steps):progress(phase='prepare shared input/teacher cache',step=i+1,total_steps=steps)
    for value in cache.values():value.flush()
    del cache
    if steps==5000:
        for name in TEACHERS:
            state=torch.load(teacher_paths[name],map_location='cpu',weights_only=False)
            assert state['input_stream_digest']==input_digest
            assert torch.equal(state['data_rng'],rng.get_state())
    torch.save(source_rng,folder/'source-input-rng.pt')
    evaluation=folder/'evaluation';evaluation.mkdir();native_panels=0;native_fields=0;energies={}
    for name,m in models.items():
        dest=evaluation/name;dest.mkdir()
        with np.load(BANKS/'field.npz') as d:x,t=d['x'],d['t']
        values=predict(m,x,t)
        with np.load(OLD/'runs'/DATASET/(name+'_direct')/'diagnostics/field.npz') as saved:np.testing.assert_array_equal(values,saved['prediction']);native_fields+=1
        np.savez_compressed(dest/'field.npz',prediction=values)
        energies[name]=dict(field_energy=float(np.mean(values.astype(np.float64)**2)),panels=[])
        for clock in TIMES:
            with np.load(BANKS/f'panel_{clock:.1f}.npz') as d:x,t=d['x'],d['t']
            value,jac=jacobian(m,torch.from_numpy(x),torch.from_numpy(t));value=value.numpy();jac=jac.numpy()
            with np.load(OLD/'runs'/DATASET/(name+'_direct')/'diagnostics'/f'panel_{clock:.1f}.npz') as saved:
                np.testing.assert_array_equal(value,saved['prediction']);np.testing.assert_array_equal(jac,saved['jacobian']);native_panels+=1
            np.savez_compressed(dest/f'panel_{clock:.1f}.npz',prediction=value,jacobian=jac)
            energies[name]['panels'].append(dict(time=clock,field_energy=float(np.mean(value.astype(np.float64)**2)),jacobian_energy=float(np.mean(np.sum(jac.astype(np.float64)**2,axis=(1,2))))))
        assert signature(m)==signatures[name] and all(p.grad is None for p in m.parameters())
    cache_hashes={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    result=dict(state='COMPLETE',steps=steps,batch_size=512,training_seed=0,source_data_generator_seed=100000,
        snapshots=snapshots,teachers=teacher_paths,teacher_signatures=signatures,teacher_energy=energies,
        old_native_fields_replayed=native_fields,old_native_panels_replayed=native_panels,elapsed_seconds=time.monotonic()-started,files_sha256=cache_hashes)
    write(folder/'metadata.json',result);return result
