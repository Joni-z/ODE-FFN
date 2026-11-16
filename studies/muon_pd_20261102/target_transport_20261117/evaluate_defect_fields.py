"""Unfitted analytic hidden-defect prediction on the unchanged complete model."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import time
from population_resolution import read,dump,sha,load_model,select_gpu


def main():
    p=argparse.ArgumentParser()
    for name in ('root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case');p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--gpu-uuid');p.add_argument('--witness',action='store_true')
    p.add_argument('--refined-reference',action='store_true')
    p.add_argument('--readiness',type=Path);p.add_argument('--cpu-witness',type=Path)
    a=p.parse_args();physical,uuid=(None,None)
    if a.device=='cuda':
        physical,uuid=select_gpu(a.gpu_uuid);os.environ['CUDA_VISIBLE_DEVICES']=uuid
    else:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    from common import base,ev,torch
    from gpu_resolution import exclusive_gpu
    from hidden_defect_transport import model_response,forced_model
    from field_exposure import complete_forward
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    guard=(lambda:exclusive_gpu(uuid)) if uuid else (lambda:None)
    guard();base.verify_sources();a.output.mkdir(parents=True,exist_ok=False)
    fields=read(a.root/'fields.json');portable=read(a.root/'portable.json')
    assert fields['portable_sha256']==sha(a.root/'portable.json')
    protocol=Path(__file__).resolve().parents[3]/'docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-protocol.md'
    assert fields['protocol_sha256']==sha(protocol)
    refinement=protocol.with_name('learned-defect-reference-refinement.md')
    if not a.witness:
        assert a.readiness and read(a.readiness)['passed']
        assert read(a.readiness)['fields_sha256']==sha(a.root/'fields.json')
        assert bool(read(a.readiness).get('refined_reference'))==a.refined_reference
    cases=[r for r in portable['cases'] if not a.case or r['id']==a.case]
    assert len(cases)==(1 if a.case else 4)
    rows=[];began=time.monotonic()

    def compare(x,y,atol,rtol):
        x,y=(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v) for v in (x,y))
        err=np.abs(x-y);ok=err<=atol+rtol*np.abs(y)
        return dict(qualified=bool(ok.all()),max_error=float(err.max()),failed_components=int((~ok).sum()))

    for case in cases:
        cpu,crossed_cpu=load_model(a.root,case)
        native,crossed=(copy.deepcopy(m).to(a.device) for m in (cpu,crossed_cpu))
        snapshots={k:ev.tensor_snapshot(v) for k,v in [('cpu',cpu),('native',native),('crossed',crossed)]}
        methods=[v.solve.__func__ for _,v in native.ode_layers()]
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if uuid else None
        for split in ('selection','confirmation'):
            row=next(r for r in fields['rows'] if (r['dataset'],r['split'])==(case['dataset'],split))
            file=a.root/row['file'];assert sha(file)==row['sha256']
            with np.load(file) as z:inputs={k:z[k].copy() for k in z.files}
            assert {k:ev.array_hash(v) for k,v in inputs.items()}==row['arrays']
            count=16 if a.witness else 4096
            x,t,oracle=(torch.from_numpy(inputs['field_'+k][:count]).double().to(a.device) for k in ('x','t','oracle'))
            folder=a.output/(case['id']+'-'+split);folder.mkdir()
            def progress(stage):
                dump(a.output/'progress.json',dict(pid=os.getpid(),case=case['id'],split=split,
                    stage=stage,points=count,completed_rows=len(rows),seconds=time.monotonic()-began))
                if uuid:
                    status=subprocess.check_output(['nvidia-smi','-i',uuid,
                        '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw',
                        '--format=csv,noheader'],text=True)
                    with (a.output/'gpu-history.jsonl').open('a') as log:
                        import json
                        log.write(json.dumps(dict(unix_time=time.time(),case=case['id'],split=split,
                            stage=stage,resource=status.strip()))+'\n')
            progress('native-crossed-field')
            with torch.no_grad():v0,v1=native(x,t),crossed(x,t)
            values=dict(x=x.cpu().numpy(),time=t.cpu().numpy(),oracle=oracle.cpu().numpy(),
                        native=v0.cpu().numpy(),crossed=v1.cpu().numpy())
            refinements={}
            for steps in ((16,32,64,128,256) if a.refined_reference else (16,32,64)):
                guard();progress('analytic-observer-'+str(steps))
                primal,tangent=model_response(native,x,t,steps,case['trained_solver'],case['inference_solver'])
                refinements[steps]=(primal,tangent)
                values['primal'+str(steps)]=primal.cpu().numpy();values['tangent'+str(steps)]=tangent.cpu().numpy()
            checks={}
            for kind,index,atol,rtol in [('primal',0,1e-7,1e-7),('tangent',1,1e-8,1e-5)]:
                checks[kind+'_refinement']=compare(refinements[32][index],refinements[64][index],atol,rtol)
                before=float((refinements[32][index]-refinements[16][index]).abs().max())
                after=float((refinements[64][index]-refinements[32][index]).abs().max())
                checks[kind+'_decreasing']=dict(qualified=after<before or max(before,after)<=1e-12,
                    difference16to32=before,difference32to64=after)
            progress('independent-continuous-prefix')
            for mode in ('continuous_loose','continuous_tight'):
                complete_forward(cpu,values['x'][:16],values['time'][:16],mode,folder/(mode+'.npz'),folder/'reference-progress.json')
            with np.load(folder/'continuous_loose.npz') as z:loose=z['predictions'].copy()
            with np.load(folder/'continuous_tight.npz') as z:tight=z['predictions'].copy()
            checks['adaptive_reference']=compare(loose,tight,1e-9,1e-9)
            checks['primal64_reference']=compare(values['primal64'][:16],tight,1e-7,1e-7)
            refined_checks={}
            if a.refined_reference:
                for kind,index,atol,rtol in [('primal',0,1e-7,1e-7),('tangent',1,1e-8,1e-5)]:
                    refined_checks[kind+'_refinement']=compare(refinements[128][index],refinements[256][index],atol,rtol)
                    before=float((refinements[128][index]-refinements[64][index]).abs().max())
                    after=float((refinements[256][index]-refinements[128][index]).abs().max())
                    refined_checks[kind+'_decreasing']=dict(qualified=after<before or max(before,after)<=1e-12,
                        difference64to128=before,difference128to256=after)
                refined_checks['primal256_reference']=compare(values['primal256'][:16],tight,1e-7,1e-7)
            progress('independent-whole-network-derivative')
            xx,tt=x[:8],t[:8];e=torch.tensor(0.,dtype=torch.float64,device=a.device)
            with torch.no_grad():
                pp,zz=torch.func.jvp(lambda eps:forced_model(native,xx,tt,16,eps,
                    case['trained_solver'],case['inference_solver']),(e,),(torch.ones_like(e),))
                checks['jvp_primal']=compare(pp,refinements[16][0][:8],1e-11,1e-11)
                checks['jvp_tangent']=compare(zz,refinements[16][1][:8],1e-11,1e-11)
                for epsilon in (1e-4,1e-5):
                    ep=torch.tensor(epsilon,dtype=torch.float64,device=a.device)
                    plus=forced_model(native,xx,tt,16,ep,case['trained_solver'],case['inference_solver'])
                    minus=forced_model(native,xx,tt,16,-ep,case['trained_solver'],case['inference_solver'])
                    fd=(plus-minus)/(2*ep);values['finite_difference'+str(epsilon)]=fd.cpu().numpy()
                    checks['finite_difference'+str(epsilon)]=compare(refinements[16][1][:8],fd,2e-7,2e-4)
            _,zero=model_response(native,xx,tt,16,case['trained_solver'],case['trained_solver'])
            reverse_p,reverse_z=model_response(native,xx,tt,16,case['inference_solver'],case['trained_solver'])
            checks['zero']=compare(zero,torch.zeros_like(zero),1e-12,0.)
            checks['reverse_primal']=compare(reverse_p,refinements[16][0][:8],1e-12,0.)
            checks['reverse_tangent']=compare(reverse_z,-refinements[16][1][:8],1e-12,0.)
            if a.cpu_witness:
                expected=a.cpu_witness/(case['id']+'-'+split)/'arrays.npz'
                with np.load(expected) as z:
                    checks['cpu_primal']=compare(values['primal16'][:16],z['primal16'],1e-9,1e-8)
                    checks['cpu_tangent']=compare(values['tangent16'][:16],z['tangent16'],1e-9,1e-8)
                    if a.refined_reference:
                        refined_checks['cpu_primal256']=compare(values['primal256'][:16],z['primal256'],1e-9,1e-8)
                        refined_checks['cpu_tangent256']=compare(values['tangent256'][:16],z['tangent256'],1e-9,1e-8)
            with torch.no_grad():assert torch.equal(v0,native(x,t))
            assert methods==[v.solve.__func__ for _,v in native.ode_layers()]
            np.savez_compressed(folder/'arrays.npz',**values)
            final_steps=256 if a.refined_reference else 64
            signal=values['crossed']-values['native'];pred=values['tangent'+str(final_steps)];energy=float(np.mean(signal**2))
            error=float(np.mean((pred-signal)**2));residual=values['native']-values['oracle']
            actual_risk=float(np.mean((values['crossed']-values['oracle'])**2)-np.mean(residual**2))
            predicted_risk=float(2*np.mean(residual*pred)+np.mean(pred**2))
            original_qualified=all(c['qualified'] for c in checks.values())
            implementation={k:v for k,v in checks.items() if k not in
                ('primal_refinement','tangent_refinement','primal_decreasing','tangent_decreasing','primal64_reference')}
            refined_qualified=all(c['qualified'] for c in refined_checks.values()) and all(c['qualified'] for c in implementation.values())
            result=dict(case=case['id'],split=split,points=count,
                numerically_qualified=refined_qualified if a.refined_reference else original_qualified,
                original_numerically_qualified=original_qualified,refined_reference=a.refined_reference,
                final_reference_steps=final_steps,checks=checks,refined_checks=refined_checks,
                signal_mse=energy,prediction_error_mse=error,score=1-error/energy if energy else None,
                actual_oracle_risk_change=actual_risk,predicted_oracle_risk_change=predicted_risk,
                file=str((folder/'arrays.npz').relative_to(a.output)),sha256=sha(folder/'arrays.npz'))
            rows.append(result);dump(a.output/'rows.json',rows)
            print('LEARNED_DEFECT_FIELD_ROW',case['id'],split,result['numerically_qualified'],flush=True);guard()
        for key,model in [('cpu',cpu),('native',native),('crossed',crossed)]:ev.same_tensors(model,snapshots[key])
        assert torch.equal(rng,torch.get_rng_state())
        if uuid:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    dump(a.output/'COMPLETE.json',dict(passed=True,witness=a.witness,rows=len(rows),
        all_numerically_qualified=all(r['numerically_qualified'] for r in rows),physical_gpu=physical,gpu_uuid=uuid,
        fields_sha256=sha(a.root/'fields.json'),portable_sha256=sha(a.root/'portable.json'),source_sha256=sha(__file__),
        refined_reference=a.refined_reference,refinement_protocol_sha256=sha(refinement) if a.refined_reference else None,
        tensors_methods_rng_unchanged=True,seconds=time.monotonic()-began))
    print('LEARNED_DEFECT_FIELD_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
