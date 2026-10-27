"""Replay the independent coefficient screen and its fixed ordinary controls."""
from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True);args = parser.parse_args()
root, out = args.root.resolve(), args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0, str(root/'source'));import jet_study as formal
started = time.monotonic();settings, geometry, native, core, operator = formal.setup(root)
formal.verify_inputs(root);assert formal.load(root/'output/status.json')['state']=='COMPLETE'
original = {str(p): native.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
basis = np.load(root/'basis.npy');reference = Path(settings['geometry_reference'])
comparisons = [];curves = [];spectra = [];clocks = [];coefficient_checks = []
monitor = [__import__('json').loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()]
launches = [__import__('json').loads(line) for line in (root/'output/launches.jsonl').read_text().splitlines()]
memory = {}
for job in launches:
    samples = [p['rss_bytes'] for row in monitor for p in row['processes'] if p['pid']==job['pid']]
    assert samples
    memory[(job['law'], job['arm'])] = max(samples)/2**30
counts = dict(new_checkpoints=0, reference_checkpoints=0, new_panels=0, reference_panels=0,
              new_samples=0, reference_samples=0, stream_batches=0, new_P_matrices=0)


def close(value, expected):
    np.testing.assert_allclose(value, expected, atol=1e-9, rtol=1e-7)


def table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def field_metrics(prediction, bank):
    error = prediction.astype(np.float64)-bank['oracle'];plane = error@basis
    normal = error-plane@basis.T
    clean = bank['x']+np.maximum(1-bank['t'], .05)*prediction.astype(np.float64)
    return dict(oracle_mse=float(np.mean(error**2)), pair_mse=float(np.mean((prediction-bank['pair'])**2)),
        prediction_rms=float(np.sqrt(np.mean(prediction**2))),
        tangent_mse_per_dimension=float(np.mean(plane**2)),
        normal_mse_per_dimension=float(np.sum(normal**2)/(len(error)*126)),
        clean_oracle_mse=float(np.mean((clean-bank['clean'])**2)))


def replay_field(model, field):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(field['x'][i:i+1024]),
            torch.from_numpy(field['t'][i:i+1024])).numpy() for i in range(0, len(field['x']), 1024)])


def local_check(layer,hidden):
    test=copy.deepcopy(layer).double();x=hidden.double();p=test.proj.weight.detach();parameter=test.coefficient.detach()
    with torch.no_grad():
        pre=torch.nn.functional.linear(x,test.t_func[0].weight,test.t_func[0].bias)
        h=torch.nn.functional.silu(pre);sig=torch.sigmoid(pre);dh=sig+pre*sig*(1-sig)
        w2=test.t_func[2].weight;logit=torch.nn.functional.linear(h,w2,test.t_func[2].bias)
        probability=torch.sigmoid(logit/10);s=.2+.8*probability
        ds=((dh*w2)@test.t_func[0].weight)*(.8/10)*probability*(1-probability)
        r=s;dr=ds
        if test.coefficient_mode=='free':
            probability_r=torch.sigmoid(torch.nn.functional.linear(h,test.free_weight,test.free_bias)/10)
            r=.2+.8*probability_r
            dr=((dh*test.free_weight)@test.t_func[0].weight)*(.8/10)*probability_r*(1-probability_r)
        u=x@p.T;w=x@(p@p).T;v=x@parameter.T;eye=torch.eye(64,dtype=x.dtype)
        if test.coefficient_mode=='direct':
            y=x+s*u+.5*s*s*v;matrix=eye+s[:,:,None]*p+.5*s.square()[:,:,None]*parameter
            j=matrix+(u+s*v)[:,:,None]*ds[:,None,:];factor=.5*s*s
        else:
            y=x+s*u+.5*s*s*w;matrix=eye+s[:,:,None]*p+.5*s.square()[:,:,None]*(p@p)
            vector=u+s*w
            if test.coefficient_mode=='quadratic':
                factor=.5*s*s;y=y+factor*v;matrix=matrix+factor[:,:,None]*parameter
                j=matrix+(vector+s*v)[:,:,None]*ds[:,None,:]
            elif test.coefficient_mode=='affine':
                factor=.3*s;y=y+factor*v;matrix=matrix+factor[:,:,None]*parameter
                j=matrix+(vector+.3*v)[:,:,None]*ds[:,None,:]
            else:
                factor=.5*s*r;y=y+factor*v;matrix=matrix+factor[:,:,None]*parameter
                j=matrix+(vector+.5*r*v)[:,:,None]*ds[:,None,:]+(.5*s*v)[:,:,None]*dr[:,None,:]
    actual=test(x)
    jacobian=torch.func.vmap(torch.func.jacrev(lambda row:test(row[None])[0]))(x)
    torch.testing.assert_close(actual,y,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(jacobian,j,atol=1e-10,rtol=1e-8)
    q=torch.linspace(-.1,.1,64,dtype=x.dtype).expand_as(x)
    gp,gc=torch.autograd.grad((test(x)*q).sum(),(test.proj.weight,test.coefficient))
    expected_p=torch.einsum('ni,nj->ij',s*q,x)
    if test.coefficient_mode!='direct':
        expected_p=expected_p+torch.einsum('ni,nj->ij',.5*s*s*q,u)+torch.einsum('ni,nj->ij',.5*s*s*(q@p),x)
    expected_c=torch.einsum('ni,nj->ij',factor*q,x)
    torch.testing.assert_close(gp,expected_p,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(gc,expected_c,atol=1e-10,rtol=1e-8)
    return dict(probes=len(x),dense_value_max=float((actual-y).detach().abs().max()),
        input_Jacobian_max=float((jacobian-j).detach().abs().max()),
        P_gradient_max=float((gp-expected_p).detach().abs().max()),
        coefficient_gradient_max=float((gc-expected_c).detach().abs().max()))


for entry in settings['laws']:
    law = entry['law'];banks = Path(entry['banks']);bank = geometry.training_bank(entry, native)
    with np.load(banks/'field.npz') as data:field = {k: data[k] for k in data.files}
    rng = torch.Generator().manual_seed(100000);digest = native.EMPTY
    digests = {0:digest};rng_states = {0:rng.get_state().clone()}
    for step in range(1, 5001):
        _, t, target, source = native.draw(bank, rng)
        digest = native.legacy.fingerprint_batch(digest, target, source, t)
        if step in geometry.STEPS:digests[step] = digest;rng_states[step] = rng.get_state().clone()
    counts['stream_batches'] += 5000
    old_zero = torch.load(reference/'output/runs'/law/'ode_k2_clean/step_0000000.pt', weights_only=False)
    for arm in (*formal.ARMS, 'ode_k2_clean', 'swiglu192_clean'):
        fresh = arm in formal.ARMS;family = arm.rsplit('_', 1)[0]
        folder = (root if fresh else reference)/'output/runs'/law/arm
        assert formal.load(folder/'status.json')['state']=='COMPLETE'
        records = [__import__('json').loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        model = formal.build(root, law, arm) if fresh else core.CappedHead(native.old_build(128, family, 'direct').raw, 'clean')
        for step in geometry.STEPS:
            state = torch.load(folder/f'step_{step:07d}.pt', weights_only=False)
            native.restore_model(model, state);model.eval()
            assert state['input_stream_digest']==digests[step]
            assert torch.equal(state['data_rng'], rng_states[step])
            if step==0:
                for key, value in old_zero['model'].items():
                    if fresh and family in ('jet_c','affine_c','free_c','direct_b') or '.ffn.' not in key:
                        assert torch.equal(value, state['model'][key]), key
                for key, value in old_zero['runtime_buffers'].items():assert torch.equal(value, state['runtime_buffers'][key]), key
                assert not state['optimizer']['state']
            else:
                identifiers = state['optimizer']['param_groups'][0]['params']
                assert len(identifiers)==len(list(model.parameters()))
                for parameter, index in zip(model.parameters(), identifiers):
                    moment = state['optimizer']['state'][index];assert int(moment['step'])==step
                    for key in ('exp_avg', 'exp_avg_sq'):
                        assert parameter.shape==moment[key].shape and torch.isfinite(moment[key]).all()
            prediction = replay_field(model, field)
            with np.load(folder/f'field_{step:07d}.npz') as data:np.testing.assert_array_equal(prediction, data['prediction'])
            metrics = field_metrics(prediction, field);stored = next(r for r in records if r['step']==step)
            for key, value in metrics.items():close(value, stored[key])
            curves.append(dict(law=law, family=family, step=step, **metrics))
            counts['new_checkpoints' if fresh else 'reference_checkpoints'] += 1
            if fresh:
                old_P = formal.load(folder/f'generator_{step:07d}.json')
                layers=list(model.ode_layers());assert len(old_P)==len(layers)
                for saved,(name,layer) in zip(old_P,layers):
                    assert saved['layer']==name
                    p=layer.proj.weight.detach().double().numpy();parameter=layer.coefficient.detach().double().numpy()
                    b=parameter if layer.coefficient_mode=='direct' else p@p+parameter
                    c=b-p@p if layer.coefficient_mode=='direct' else parameter
                    arrays={'P_singular_values':np.linalg.svd(p,compute_uv=False),
                            'reference_B_singular_values':np.linalg.svd(b,compute_uv=False),
                            'C_singular_values':np.linalg.svd(c,compute_uv=False)}
                    for key,value in arrays.items(): np.testing.assert_allclose(value,saved[key],atol=1e-10,rtol=1e-8)
                    for key,value in dict(P_frobenius=np.linalg.norm(p),reference_B_frobenius=np.linalg.norm(b),
                        C_frobenius=np.linalg.norm(c),P_C_commutator_frobenius=np.linalg.norm(p@c-c@p)).items():close(value,saved[key])
                    counts['new_P_matrices']+=1
                    spectra.append(dict(law=law,family=family,step=step,layer=name,
                        sigma_max=float(arrays['P_singular_values'][0]),sigma_min=float(arrays['P_singular_values'][-1]),
                        singular_cv=float(np.std(arrays['P_singular_values'])/np.mean(arrays['P_singular_values'])),
                        C_frobenius=float(np.linalg.norm(c)),reference_B_frobenius=float(np.linalg.norm(b)),
                        P_C_commutator_frobenius=float(np.linalg.norm(p@c-c@p))))
                saved_visits = formal.load(folder/f'clocks_{step:07d}.json')
                actual_visits = native.legacy.visit_diagnostics(model, banks)
                assert len(saved_visits)==len(actual_visits)
                for saved, actual in zip(saved_visits, actual_visits):
                    assert saved['layer']==actual['layer'] and saved['visit']==actual['visit']
                    assert saved['activation'].keys()==actual['activation'].keys()
                    for key, value in actual['activation'].items():
                        if isinstance(value, (float, int)):close(value, saved['activation'][key])
                        else:assert value==saved['activation'][key]
                    clocks.append(dict(law=law, family=family, step=step, layer=actual['layer'], **actual['activation']))
            if fresh and family!='swiglu258':
                captured=[];hooks=[]
                for name,layer in model.ode_layers():
                    def capture(module,inputs,output,key=name):captured.append((key,module,inputs[0].detach().clone()))
                    hooks.append(layer.register_forward_hook(capture))
                with np.load(banks/'probe.npz') as data:
                    with torch.no_grad():model(torch.from_numpy(data['x'][:32]),torch.from_numpy(data['t'][:32]))
                for hook in hooks:hook.remove()
                assert len(captured)==6
                for name,layer,hidden in captured:
                    coefficient_checks.append(dict(law=law,family=family,step=step,layer=name,**local_check(layer,hidden)))
        result = formal.load(folder/'result.json');cost = native.compute_budget(model)
        for key in ('parameters', 'dense_forward_macs', 'dense_training_forward_backward_macs'):assert cost[key]==result[key]
        tangent_J=[]
        for clock in geometry.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x, t, truth = data['x'], data['t'], data['oracle_jacobian']
            value, jacobian = native.legacy.jacobian(model, torch.from_numpy(x), torch.from_numpy(t))
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(value.numpy(), data['prediction']);np.testing.assert_array_equal(jacobian.numpy(), data['jacobian'])
            stored = next(r for r in result['panels'] if abs(r['time']-clock)<1e-6)
            close(np.mean(np.sum((jacobian.numpy()-truth)**2, axis=(1, 2))), stored['jacobian_error'])
            plane_error=np.einsum('ai,nij->naj',basis.T,jacobian.numpy()-truth)
            tangent_J.append(float(np.mean(np.sum(plane_error**2,axis=(1,2)))))
            counts['new_panels' if fresh else 'reference_panels'] += 1
        with np.load(banks/'sampling.npz') as data:source, target, directions = [torch.from_numpy(data[k]) for k in ('source', 'target', 'directions')]
        generated = native.legacy.generated_samples(model, source, 32)
        with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(generated.numpy(), data['generated'])
        close(native.legacy.sliced_w2(generated, target, directions), result['sw2'])
        angles = torch.arange(128)*2*torch.pi/128;td = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
        close(native.legacy.sliced_w2(torch.tensor(generated.numpy()@basis, dtype=torch.float32),
            torch.tensor(target.numpy()@basis, dtype=torch.float32), td), result['tangent_sw2'])
        counts['new_samples' if fresh else 'reference_samples'] += 1
        comparisons.append(dict(law=law,family=family,new_training=fresh,tangent_jacobian_error=float(np.mean(tangent_J)),
            cell_seconds=float(formal.load(folder/'status.json')['elapsed_seconds']),
            peak_sampled_rss_gib=memory[(law, arm)] if fresh else None,
            **{key: result[key] for key in ('oracle_mse', 'tangent_mse_per_dimension', 'normal_mse_per_dimension',
                'jacobian_error', 'sw2', 'tangent_sw2', 'normal_sample_energy_per_dimension',
                'parameters', 'dense_forward_macs', 'dense_training_forward_backward_macs')}))
        print(dict(law=law, family=family, counts=counts, elapsed=time.monotonic()-started), flush=True)

expected=dict(new_checkpoints=40,reference_checkpoints=16,new_panels=50,reference_panels=20,
    new_samples=10,reference_samples=4,stream_batches=10000,new_P_matrices=192)
assert counts==expected,counts
assert len(coefficient_checks)==192
checks=[];eligible={};scores={}
selected={law:{r['family']:r for r in comparisons if r['law']==law} for law in ('thin','thick')}
for candidate in ('jet_c','direct_b'):
    for law in ('thin','thick'):
        row=selected[law][candidate];ordinary=[selected[law][f] for f in ('swiglu192','swiglu258')]
        for key in ('oracle_mse','tangent_mse_per_dimension','jacobian_error','tangent_jacobian_error','sw2','tangent_sw2'):
            threshold=(.9 if law=='thin' else 1.05) if key in ('oracle_mse','tangent_mse_per_dimension') else 1.10
            ratio=row[key]/min(v[key] for v in ordinary)
            checks.append(dict(candidate=candidate,law=law,control='better_Swi',metric=key,ratio=ratio,
                               threshold=threshold,passed=ratio<=threshold))
    for control in ('affine_c','free_c'):
        ratio=float(np.sqrt(np.prod([selected[law][candidate]['tangent_mse_per_dimension']/
                                     selected[law][control]['tangent_mse_per_dimension'] for law in ('thin','thick')])))
        checks.append(dict(candidate=candidate,law='geometric_mean',control=control,metric='tangent_mse_per_dimension',
                           ratio=ratio,threshold=.95,passed=ratio<=.95))
    eligible[candidate]=all(r['passed'] for r in checks if r['candidate']==candidate)
    scores[candidate]=float(np.sqrt(np.prod([selected[law][candidate]['tangent_mse_per_dimension']/min(
        selected[law][f]['tangent_mse_per_dimension'] for f in ('swiglu192','swiglu258')) for law in ('thin','thick')])))
nominee=None
if eligible['jet_c'] and eligible['direct_b']:nominee='direct_b' if scores['direct_b']<=1.05*scores['jet_c'] else 'jet_c'
elif eligible['jet_c']:nominee='jet_c'
elif eligible['direct_b']:nominee='direct_b'
formal.verify_inputs(root)
for name,digest in original.items():assert native.sha(Path(name))==digest,name
for name,rows in [('comparison.csv',comparisons),('curves.csv',curves),('P-history.csv',spectra),
                  ('clocks.csv',clocks),('coefficient-checks.csv',coefficient_checks),('decision-checks.csv',checks)]:table(out/name,rows)
formal.write(out/'decision.json',dict(context_lead=nominee is not None,eligible=eligible,scores=scores,nominee=nominee,
    decision='schedule_historical_confirmation' if nominee else 'stop_this_coefficient_setting',
    new_architecture_nominated=False,no_degree_rank_clock_coefficient_initializer_rescue=True))
formal.write(out/'audit.json',dict(status='PASS',counts=counts,formal_files_unchanged=len(original),
    elapsed_seconds=time.monotonic()-started,fitted_dense_coefficient_checks=len(coefficient_checks),
    fitted_local_input_jacobian_anchors=192*32,
    peak_sampled_total_rss_gib=max(sum(p['rss_bytes'] for p in row['processes']) for row in monitor)/2**30,
    scope='native checkpoint/Jacobian/sample replay, streams/optimizer/P/B/C/clocks, independent local values/input/parameter gradients and fixed selection'))
shutil.copyfile(root/'preflight.json',out/'preflight.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in comparisons:
        if row['new_training']:
            keep=row['family']==nominee
            stream.write(f"{settings['registration_commit'][:7]}\t{row['oracle_mse']:.10g}\t{row['peak_sampled_rss_gib']:.3f}\t{'keep' if keep else 'discard'}\t{row['law']} {row['family']} clean; {'context lead only' if keep else 'fixed screen, evidence retained'}\n")
shutil.copyfile(root/'results.tsv',out/'results.tsv')
print(dict(status='AUDIT_PASS',eligible=eligible,nominee=nominee,counts=counts,elapsed=time.monotonic()-started),flush=True)
