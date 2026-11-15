"""Evaluate the state-flow pair and check frozen-weight solver refinement."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from stage_trial import HERE,TAG,base,torch,build_model
import stage_evaluate as evaluation

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--numpy-python',required=True)
    a=p.parse_args();root=Path(a.root);out=root/'stage-analysis';out.mkdir(exist_ok=False);torch.set_num_threads(1)
    def phase(name,**kw):base.dump(out/'status.json',dict(phase=name,pid=os.getpid(),unix_time=time.time(),**kw))
    try:
        terminal=root/'stage-output/monitor/queue-finished.json'
        while not terminal.exists():
            s=json.loads((root/'stage-output/monitor/status.json').read_text())
            try:os.kill(s['queue_pid'],0)
            except ProcessLookupError:
                if not terminal.exists():raise
            phase('waiting_for_training',queue_pid=s['queue_pid']);time.sleep(20)
        done=json.loads(terminal.read_text());assert not done['failed'] and not done['unstarted'] and len(done['completed'])==2
        grid=json.loads((HERE/'stage-grid.json').read_text());completed=0;refined=[]
        for split,steps in [('selection',['15000','20000']),('confirmation',['20000'])]:
            for c in grid['cells']:
                folder=root/'stage-output'/TAG/c['dataset']/c['id']/'seed000'
                phase('evaluating',split=split,case=c['id'],completed=completed,total=4)
                cmd=[sys.executable,str(HERE/'stage_evaluate.py'),'--folder',str(folder),'--data-root',a.data_root,'--assets',str(root/'assets'),
                     '--output',str(out/split/c['id']),'--device','cpu','--split',split,'--steps',*steps]
                with (out/(split+'-'+c['id']+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                completed+=1
        for c in grid['cells']:
            phase('internal_refinement',case=c['id'])
            folder=root/'stage-output'/TAG/c['dataset']/c['id']/'seed000';cfg=json.loads((folder/'config.json').read_text());cache=base.verify_data(a.data_root,c)
            model,identity=evaluation.evaluator.ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu')
            arrays,fixed,binding=evaluation.evaluator.assets(root/'assets',c['dataset'],'confirmation','cpu')
            snapshot=evaluation.evaluator.ev.tensor_snapshot(model);predictions={};samples={};scores=[]
            for n in [4,8,16]:
                for _,layer in model.ode_layers():layer.substeps=n
                before=time.monotonic()
                prediction,mse=evaluation.evaluator.ev.field(model,fixed)
                source=torch.from_numpy(arrays['source_0'][:512]);target=torch.from_numpy(arrays['target_0'][:512])
                value=base.train.generated_samples(model,source,128)
                sw=base.train.sliced_w2(value.double(),target.double(),fixed['directions'].double())
                predictions[n]=torch.from_numpy(prediction);samples[n]=value
                scores.append(dict(internal_substeps=n,field_mse=mse,one_bank_512_sw2=sw,seconds=time.monotonic()-before,
                     prediction_shift_from4=0. if n==4 else float((predictions[n]-predictions[4]).square().mean().sqrt()),
                     sample_shift_from4=0. if n==4 else float((samples[n]-samples[4]).square().mean().sqrt())))
            evaluation.evaluator.ev.same_tensors(model,snapshot)
            refined.append(dict(id=c['id'],checkpoint=identity,assets_sha256=binding['sha256'],rows=scores))
            base.dump(out/'refinement.json',dict(passed=True,scope='bounded512-sample solver test, not pooled full-bank ranking',cases=refined))
        phase('independent_recomputation')
        code=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from analyze_screen import collect;'
              'r=Path(sys.argv[2]);collect(r,r,split="selection",expected_cases=2);collect(r,r,split="confirmation",expected_cases=2)')
        subprocess.run([a.numpy_python,'-c',code,str(HERE.parent/'response_screen_20261115'),str(out)],check=True)
        base.dump(out/'COMPLETE.json',dict(passed=True,training_cases=2,selection_cases=2,confirmation_cases=2,refinement_cases=2,source_sha256=base.digest(__file__)))
        phase('complete',completed=completed,total=4)
    except Exception as e:phase('failed',error=repr(e));raise

if __name__=='__main__':main()
