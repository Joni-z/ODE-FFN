"""Verify public evidence and preserve the successful native analysis once."""
import argparse,ast,copy,csv,importlib.util,json,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np

def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
    assert not (root/'sealed-runtime.json').exists();study.verify(root)
    assert study.load(root/'controller/status.json')['state']=='COMPLETE'
    phases=study.load(root/'controller/launches.json');children=study.load(root/'output/exits.json')
    assert len(phases)==2 and len(children)==8 and all(r['exit_code']==0 for r in [*phases,*children])
    assert study.load(root/'controller-tool-exit.json')==dict(session=25342,exit_code=0)
    assert study.load(root/'curation-tool-exit.json')['exit_code']==0
    receipt=study.load(root/'documented-pass.json');assert receipt['status']=='PASS' and receipt['exit_code']==0
    assert receipt['exec_session']==21100 and not receipt['doc_vs_reality_mismatches']
    assert receipt['preflight_sha256']==study.sha(root/'preflight.json')
    for name in ('study.py','feedback_math.py','audit.py','controller.py'):
        assert study.sha(doc/name)==study.sha(root/'source'/name)
    registration=study.load(root/'curation-source/registration.json')
    assert study.sha(root/'curation-source/curate.py')==study.sha(doc/'curate.py')==registration['sha256']
    for p in (root/'curation').rglob('*'):
        if p.is_file():assert study.sha(p)==study.sha(doc/'results'/p.relative_to(root/'curation'))
    read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
    rows=read('groups.csv');variance=read('clock-variance.csv');cases=read('fixed-cases.csv')
    assert len(rows)==480 and len(variance)==48 and len(cases)==40
    audit=study.load(root/'audit/result.json');assert audit['status']=='PASS' and len(audit['rows'])==480
    assert audit['probes']==122880 and audit['native_endpoints_bitwise']==8
    spec=importlib.util.spec_from_file_location('registered_feedback_curator',root/'curation-source/curate.py')
    curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    # Compare every matrix entry as well as the already-audited spectra. A
    # spectrum alone would not catch a mistaken rotation of a derivative.
    import torch,audit as independent,feedback_math
    assert Path(independent.__file__).resolve()==root/'source/audit.py'
    torch.set_num_threads(1);last_cell=None;matrix_error=0.;matrix_probes=0
    for row in rows:
        arm=row['family'];step=int(row['step']);key=f"{row['domain']}-t{int(row['time_index']):03d}-{row['layer']}"
        cell=f'{arm}-{step}';folder=root/'output'/cell
        with np.load(folder/f'{key}-spectra.npz') as f:arrays={k:f[k] for k in f.files}
        for measure,values in curator.describe(arrays).items():
            for name,value in curator.stats(values).items():
                actual=row[measure+'_'+name]
                if value is None:assert not actual
                else:assert float(actual)==value,(key,measure,name)
        if last_cell!=cell:model=study.native_model(cell);last_cell=cell
        module=copy.deepcopy(study.module_for(model,key)).double().eval().requires_grad_(False)
        with np.load(folder/f'{key}-capture.npz') as f:x=torch.from_numpy(f['h']).double()
        local=row['kind']=='local';fn=independent.layer if local else independent.core
        with torch.no_grad():analytic=feedback_math.local(module,x) if local else feedback_math.core(module,x)
        observed=torch.func.vmap(torch.func.jacrev(lambda h:fn(module,h)))(x)
        torch.testing.assert_close(analytic['J'],observed,rtol=1e-8,atol=1e-10)
        matrix_error=max(matrix_error,float((analytic['J']-observed).abs().max()))
        if 'R' in analytic:
            held=independent.clock(module,x).detach() if local else independent.held_values(module,x)
            reference=torch.func.vmap(torch.func.jacrev(lambda h,c:fn(module,h,c),argnums=0))(x,held)
            torch.testing.assert_close(analytic['R'],reference,rtol=1e-8,atol=1e-10)
            matrix_error=max(matrix_error,float((analytic['R']-reference).abs().max()))
        matrix_probes+=len(x)
        if matrix_probes%15360==0:print(json.dumps(dict(phase='matrix_entry_verification',probes=matrix_probes,max_error=matrix_error)),flush=True)
    # Independent balanced-variance reconstruction from the native scalar arrays.
    total_values=0
    for row in variance:
        cell=f"{row['family']}-{row['step']}";arrays=[]
        for j in study.INDICES:
            with np.load(root/'output'/cell/f"{row['domain']}-t{j:03d}-{row['layer']}-capture.npz") as f:arrays.append(f['s_all'].astype(np.float64))
        x=np.stack(arrays);total_values+=x.size
        mean=x.mean();t=x.mean(axis=2).mean(axis=1);image=x.mean(axis=2)
        expected=dict(variance_total=np.mean(x*x)-mean*mean,variance_time=np.var(t),
            variance_image=np.mean(np.var(image,axis=1)),variance_token=np.mean(np.var(x,axis=2)))
        for key,value in expected.items():assert abs(float(row[key])-value)<1e-12
        assert abs(sum(expected[k] for k in ('variance_time','variance_image','variance_token'))-expected['variance_total'])<1e-12
    assert total_values==983040
    assert len(read('pooled.csv'))==24 and len(read('time-profiles.csv'))==80 and len(read('variance-pooled.csv'))==8
    summary=study.load(doc/'results/summary.json');assert not summary['new_training'] and not summary['new_quality_result']
    review=study.load(root/'visual-review.json');pngs={p.name for p in (root/'curation/figures').glob('*.png')}
    assert review['status']=='PASS' and len(pngs)==3 and set(review['files'])==pngs
    for name,digest in review['files'].items():assert study.sha(root/'curation/figures'/name)==digest
    for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md'):assert (doc/name).exists()
    protected=['src','synthetic','configs','scripts','reports/report1']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
    links=0
    for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith(('/','mailto:')):continue
            resolved=(p.parent/unquote(target)).resolve();assert resolved.exists(),(p,target)
            relative=resolved.relative_to(repo)
            assert subprocess.check_output(['git','-C',str(repo),'ls-files','--',str(relative)],text=True).strip(),(p,'untracked link',target)
            links+=1
    for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    study.verify(root)
    assert matrix_probes==122880
    study.write(doc/'results/artifact-verification.json',dict(status='PASS',native_endpoints=8,all_probe_AD_checks=matrix_probes,
        complete_matrix_entry_max_error=matrix_error,
        group_summaries_checked=480,clock_values=total_values,figures_visually_reviewed=3,local_links=links,
        protected_scopes=protected,new_training=False,new_quality_result=False))
    shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
    for p in doc.iterdir():
        if p.is_file():shutil.copyfile(p,closure/p.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
    study.write(root/'sealed-runtime.json',{str(p.relative_to(root)):study.sha(p) for p in files})
    for name,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/name)==digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),
        private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
