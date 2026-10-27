"""Independent literal-Heun branches, signed accounting and QR-null checks."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True);args = parser.parse_args()
root, out = args.root.resolve(), args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0, str(root/'source'));import path_study as formal
started = time.monotonic();formal.verify_inputs(root)
settings, _, old, _, native, core = formal.setup(root)
assert formal.load(root/'output/status.json')['state']=='COMPLETE'
original = {str(p):native.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
read = lambda name:list(csv.DictReader((root/'output'/name).open()))
endpoints, step_rows, bin_rows = [read(name) for name in ('endpoints.csv', 'steps.csv', 'bins.csv')]
assert (len(endpoints), len(step_rows), len(bin_rows))==(12, 1152, 144)
basis = np.load(Path(settings['parent'])/'basis.npy');checks = [];gain_rows = []
counts = dict(native_endpoints=0, learned_steps=0, oracle_branches=0, signed_source_step_scopes=0)


@torch.no_grad()
def literal_step(field, x, i):
    t = torch.full((len(x), 1), i/32, dtype=x.dtype)
    first = field(x, t);prediction = x+(1/32)*first
    second = field(prediction, torch.full_like(t, (i+1)/32))
    following = x+(0.5/32)*(first+second)
    return following, first, prediction, second


@torch.no_grad()
def literal_tail(field, x, start):
    x = x.clone()
    for i in range(start, 32):x = literal_step(field, x, i)[0]
    return x


def parts(value):
    value = value.astype(np.float64);plane = value@basis
    return (value, plane, value-plane@basis.T)


def energies(value):
    return {name:float(np.einsum('nd,nd->', v, v)/(len(v)*d))
            for name, v, d in zip(('full','tangent','normal'), parts(value), (128,2,126))}


def close(value, text):
    if value is None:assert text==''
    else:np.testing.assert_allclose(value, float(text), atol=1e-9, rtol=1e-7)


def qr_projector(state):
    w = state['model']['raw.output.weight'].double().numpy();gamma = state['model']['raw.norm.weight'].double().numpy()
    width = len(gamma);v = np.eye(width)[:,0]-np.ones(width)/np.sqrt(width)
    center_basis = (np.eye(width)-2*np.outer(v,v)/(v@v))[:,1:]
    columns = np.column_stack(((w*gamma[None])@center_basis, basis))
    q,r = np.linalg.qr(columns,mode='reduced');left,singular,_ = np.linalg.svd(r,full_matrices=False)
    rank = int(np.sum(singular>1e-10*max(float(singular[0]),1.)))
    span = q@left[:,:rank];projector = np.eye(128)-span@span.T
    assert np.max(np.abs(columns.T@projector))<=1e-8
    c0 = w@state['model']['raw.norm.bias'].double().numpy()+state['model']['raw.output.bias'].double().numpy()
    return projector,128-rank,c0


with torch.no_grad():
    for entry in old['laws']:
        law = entry['law'];banks = Path(entry['banks'])
        with np.load(banks/'sampling.npz') as data:source = torch.from_numpy(data['source'])
        oracle = formal.oracle_for(core,basis,entry['normal_variance'])
        with np.load(root/'output/raw'/f'{law}-oracle-path.npz') as data:reference = data['path']
        np.testing.assert_array_equal(native.legacy.generated_samples(oracle,source,32).numpy(),reference[-1])
        for i in range(32):np.testing.assert_array_equal(literal_step(oracle,torch.from_numpy(reference[i]),i)[0].numpy(),reference[i+1])
        normal_probe = np.eye(128)[0]-basis@basis[0]
        probe = torch.tensor(normal_probe[None],dtype=torch.float64)
        rates = [float(np.sum(oracle(probe,torch.full((1,1),i/32,dtype=torch.float64)).numpy()[0]*normal_probe)/(normal_probe@normal_probe)) for i in range(33)]
        gains = []
        for first_index in range(33):
            value = 1.
            for i in range(first_index,32):
                velocity = rates[i]*value;predicted = value+velocity/32
                value += (velocity+rates[i+1]*predicted)/64
            gains.append(value)
        gains = np.array(gains)
        expected_gains = formal.load(root/'preflight.json')['normal_gain_'+law]
        np.testing.assert_allclose(gains,expected_gains,atol=1e-12,rtol=0)
        prefix = [1.];bias_prefix = [0.]
        for i in range(32):
            def scalar(y,t,c):return (c-y)/max(1-t,.05)
            for values,c in ((prefix,0.),(bias_prefix,1.)):
                first = scalar(values[-1],i/32,c);predicted = values[-1]+first/32
                values.append(values[-1]+(first+scalar(predicted,(i+1)/32,c))/64)
        prefix,bias_prefix = np.array(prefix),np.array(bias_prefix)
        for i in range(33):gain_rows.append(dict(law=law,index=i,time=i/32,oracle_normal_suffix=gains[i],clean_prefix=prefix[i],clean_bias_prefix=bias_prefix[i]))
        for family in formal.FAMILIES:
            model,folder,state = formal.model_and_folder(root,law,family)
            with np.load(root/'output/raw'/f'{law}-{family}.npz') as data:
                path,bridges,local,saved_signed,forced_signed,saved_projector,old_model_ids,old_oracle_ids = [data[k] for k in
                    ('path','bridges','local_defects','signed_per_source','forced_signed_per_source','forced_projector','model_mode_ids','oracle_mode_ids')]
            assert path.shape==bridges.shape==(33,2048,128) and local.shape==(32,2048,128)
            np.testing.assert_array_equal(path[0],source.numpy())
            np.testing.assert_array_equal(native.legacy.generated_samples(model,source,32).numpy(),path[-1]);counts['native_endpoints']+=1
            for i in range(33):
                replay = literal_tail(oracle,torch.from_numpy(path[i]),i)
                np.testing.assert_array_equal(replay.numpy(),bridges[i]);counts['oracle_branches']+=1
            error = bridges[-1].astype(np.float64)-bridges[0]
            delta = bridges[1:].astype(np.float64)-bridges[:-1]
            assert np.max(np.abs(delta.sum(0)-error))<=1e-10
            record = next(r for r in endpoints if r['law']==law and r['family']==family)
            selected_steps = [r for r in step_rows if r['law']==law and r['family']==family]
            projector,dimension,c0 = qr_projector(state)
            projector_difference = float(np.max(np.abs(projector-saved_projector)))
            assert projector_difference<=1e-8
            observed = bridges.astype(np.float64)@projector
            prediction = gains[:,None,None]*(prefix[:,None,None]*(source.numpy().astype(float)@projector)[None]
                +bias_prefix[:,None,None]*(c0@projector)[None,None])
            forced_max = float(np.max(np.abs(observed-prediction)));assert forced_max<=1e-4
            forced_error = error@projector
            forced_delta = delta@projector
            independent_forced = np.sum(forced_delta*forced_error[None],axis=2)/126
            np.testing.assert_allclose(independent_forced,forced_signed,atol=1e-9,rtol=1e-7)
            close(dimension,record['forced_normal_dimension']);close(np.mean(np.sum(forced_error**2,axis=1))/126,record['forced_normal_endpoint_mse'])
            for scope_index,(scope,e,d,l,dimension) in enumerate(zip(('full','tangent','normal'),parts(error),parts(delta),parts(local),(128,2,126))):
                signed = np.sum(d*e[None],axis=2)/dimension
                np.testing.assert_allclose(signed.sum(0),np.sum(e*e,axis=1)/dimension,atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(signed,saved_signed[:,:,scope_index],atol=1e-9,rtol=1e-7)
                counts['signed_source_step_scopes'] += signed.size
                mse = float(np.mean(np.sum(e*e,axis=1))/dimension)
                close(mse,record[scope+'_paired_endpoint_mse']);share = float(signed[24:].sum(0).mean()/mse) if mse>1e-20 else None
                close(share,record[scope+'_late_signed_share']);assert record[scope+'_late_aligned']==('' if share is None else str(share>=.5))
                for i in range(32):
                    row = next(r for r in selected_steps if r['scope']==scope and int(r['step'])==i)
                    local_energy = float(np.sum(l[i]**2)/(len(e)*dimension));transported = float(np.sum(d[i]**2)/(len(e)*dimension))
                    for value,key in [(signed[i].mean(),'signed_endpoint_mse_contribution'),(local_energy,'local_step_energy'),(transported,'transported_step_energy'),
                        (np.sqrt(transported/local_energy) if local_energy>1e-20 else None,'finite_rms_gain')]:close(value,row[key])
                    if scope=='normal':
                        close(independent_forced[i].mean(),row['forced_signed_contribution'])
                        close(signed[i].mean()-independent_forced[i].mean(),row['remaining_signed_contribution'])
                for group in range(4):
                    row = next(r for r in bin_rows if r['law']==law and r['family']==family and r['scope']==scope and r['time_bin']==f'{group/4:.2f}-{(group+1)/4:.2f}')
                    value = float(signed[group*8:(group+1)*8].sum(0).mean())
                    close(value,row['signed_endpoint_mse_contribution']);close(value/mse if mse>1e-20 else None,row['signed_share'])
                    close(np.sum(l[group*8:(group+1)*8]**2)/(8*len(e)*dimension),row['local_step_energy_mean'])
                    close(np.sum(d[group*8:(group+1)*8]**2)/(8*len(e)*dimension),row['transported_step_energy_mean'])
                    close(float(independent_forced[group*8:(group+1)*8].sum(0).mean()) if scope=='normal' else None,row['forced_signed_contribution'])
            normal_error = parts(error)[2];normal_local = parts(local)[2];normal_delta = parts(delta)[2]
            normal_check = float(np.max(np.abs(normal_delta-gains[1:,None,None]*normal_local)))
            total_normal_check = float(np.max(np.abs(normal_error-(gains[1:,None,None]*normal_local).sum(0))))
            assert normal_check<=1e-4 and total_normal_check<=1e-4
            close(normal_check,record['normal_propagation_max']);close(total_normal_check,record['normal_total_propagation_max'])
            close(np.mean(np.sum(forced_error**2,axis=1))/np.mean(np.sum(normal_error**2,axis=1)),record['forced_fraction_of_normal_paired_error'])
            for i in range(32):
                x = torch.from_numpy(path[i]);following,first,predicted,second = literal_step(model,x,i)
                np.testing.assert_array_equal(following.numpy(),path[i+1]);counts['learned_steps']+=1
                oracle_next,oracle_first,_,_ = literal_step(oracle,x,i)
                np.testing.assert_array_equal(following.numpy().astype(float)-oracle_next.numpy(),local[i])
                t = torch.full((2048,1),i/32,dtype=source.dtype)
                second_ref = oracle(predicted,torch.full_like(t,(i+1)/32))
                ref = torch.from_numpy(reference[i]);ref_model = model(ref,t);ref_oracle = oracle(ref,t)
                for row in [r for r in selected_steps if int(r['step'])==i]:
                    for label,value in [('model_stage1',first.numpy().astype(float)-oracle_first.numpy()),
                        ('model_stage2',second.numpy().astype(float)-second_ref.numpy()),('oracle_path_stage1',ref_model.numpy().astype(float)-ref_oracle.numpy())]:
                        for name,value in energies(value).items():close(value,row[label+'_'+name])
                    close(np.sqrt(energies(path[i])['normal']),row['model_path_normal_rms']);close(np.sqrt(energies(reference[i])['normal']),row['oracle_path_normal_rms'])
            means = 3*np.column_stack((np.cos(np.arange(8)*2*np.pi/8),np.sin(np.arange(8)*2*np.pi/8)))
            ids = [np.argmin(np.sum(((v@basis)[:,None]-means[None])**2,axis=2),axis=1) for v in (path[-1],reference[-1])]
            np.testing.assert_array_equal(ids[0],old_model_ids);np.testing.assert_array_equal(ids[1],old_oracle_ids)
            close(np.mean(ids[0]!=ids[1]),record['sourcewise_mode_disagreement'])
            close(np.sqrt(energies(path[-1])['normal']),record['model_endpoint_normal_rms'])
            close(np.sqrt(energies(reference[-1])['normal']),record['oracle_endpoint_normal_rms'])
            checks.append(dict(law=law,family=family,native_endpoint_bitwise=True,oracle_branches_bitwise=33,
                normal_propagation_max=normal_check,QR_projector_max_difference=projector_difference,forced_bridge_max=forced_max))
            assert time.monotonic()-started<600
            print(dict(checked=len(checks),law=law,family=family,elapsed=time.monotonic()-started),flush=True)

assert counts==dict(native_endpoints=12,learned_steps=384,oracle_branches=396,signed_source_step_scopes=12*32*2048*3)
formal.verify_inputs(root)
for name,digest in original.items():assert native.sha(Path(name))==digest,name
formal.table(out/'independent-checks.csv',checks);formal.table(out/'normal-gains.csv',gain_rows)
formal.write(out/'audit.json',dict(status='PASS',counts=counts,formal_files_unchanged=len(original),
    elapsed_seconds=time.monotonic()-started,new_training=False,new_sampler_nominated=False,
    scope='literal native arithmetic/branch replay, independent signed projections, oracle-probed normal rates and centered-basis QR readout null space'))
for name in ('endpoints.csv','steps.csv','bins.csv','telemetry.csv','receipt.json'):shutil.copyfile(root/'output'/name,out/name)
shutil.copyfile(root/'preflight.json',out/'preflight.json')
receipt = formal.load(root/'output/receipt.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in endpoints:stream.write(f"{settings['registration_commit'][:7]}\t{float(row['full_paired_endpoint_mse']):.10g}\t{receipt['peak_sampled_rss_gib']:.3f}\tkeep\tfrozen paired-path diagnostic {row['law']} {row['family']}; no sampler nomination\n")
shutil.copyfile(root/'results.tsv',out/'results.tsv')
print(dict(status='AUDIT_PASS',counts=counts,elapsed=time.monotonic()-started),flush=True)
