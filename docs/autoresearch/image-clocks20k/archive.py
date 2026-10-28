"""Verify public clock-study evidence and preserve one immutable closure."""
from pathlib import Path
import argparse,ast,csv,importlib.util,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np

def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
    assert not (root/'sealed-runtime.json').exists();study.verify(root,True)
    assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    phases=study.load(root/'orchestrator/launches.json');children=study.load(root/'output/exits.json')
    assert len(phases)==4 and len(children)==2 and all(r['exit_code']==0 for r in [*phases,*children])
    assert study.load(root/'tool-exit.json')==dict(session=21980,exit_code=0)
    assert study.load(root/'curation-tool-exit.json')==dict(session=4663,exit_code=0)
    preflight=study.load(root/'preflight.json');docpass=study.load(root/'documented-pass.json')
    assert preflight['status']==docpass['status']=='PASS' and preflight['checkpoint_step']==20000
    assert docpass['exec_session']==52748 and docpass['exit_code']==0 and not docpass['doc_vs_reality_mismatches']
    assert study.sha(root/'preflight.json')==docpass['preflight_sha256']
    for source,target in [('preflight.json','preflight.json'),('documented-pass.json','agent-doc-pass.json')]:
        assert study.sha(root/source)==study.sha(doc/'readiness'/target)
    for name in ('study.py','audit.py','controller.py'):assert study.sha(doc/name)==study.sha(root/'source'/name)
    registration=study.load(root/'curation-source/registration.json')
    assert study.sha(root/'curation-source/curate.py')==study.sha(doc/'curate.py')==registration['sha256']
    for phase,count in [('sampling',2),('features',9)]:
        d=study.load(root/f'audit-{phase}/result.json');assert d['status']=='PASS' and len(d['rows'])==count
    for p in (root/'curation').rglob('*'):
        if p.is_file():assert study.sha(p)==study.sha(doc/'results'/p.relative_to(root/'curation'))
    read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
    rows=read('comparison.csv');formal=study.load(root/'metric-output/result.json')['rows']
    checked=study.load(root/'audit-features/result.json')['rows'];assert len(rows)==len(formal)==len(checked)==9
    for r in rows:
        family=r['family'];actual=next(v for v in formal if v['family']==family)
        for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction','parameters','FFN_forward_MACs'):
            assert float(r[key])==actual[key]
        assert int(r['step'])==20000 and (int(r['generated']),int(r['reference']),int(r['Heun_steps']))==(1024,2048,128)
        assert abs(float(r['fid'])-next(v['fid'] for v in checked if v['family']==family))<1e-8
        assert float(r['out_of_range_fraction'])==float(np.mean(np.abs(np.load(study.sample_bank(root,family)))>1))
    calibration=read('calibration.csv');assert len(calibration)==12
    for r in calibration:
        actual=study.load(root/'calibration.json')[r['family']][r['layer']]
        for key,value in actual.items():assert float(r[key])==value
    panels=read('time-panels.csv');assert len(panels)==45
    for r in panels:
        t=float(r['time']);family=r['family']
        with np.load(study.PARENTS['native']/'banks'/f'panel_{t:.1f}.npz') as b:
            target=b['target'].astype(np.float64);delta=b['delta'].astype(np.float64)[:,None,None,None]
        p=root/'output/runs'/family.removesuffix('_constant')/f'panel-{t:.1f}.npz' if family.endswith('_constant') else study.PARENTS['budget']/'output/runs'/family/f'panel_0020000_{t:.1f}.npz'
        with np.load(p) as b:error=b['prediction'].astype(np.float64)-target
        np.testing.assert_allclose([float(r['weighted_mse']),float(r['clean_mse'])],[np.mean((error/delta)**2),np.mean(error**2)],rtol=1e-12,atol=1e-12)
    decision=study.load(doc/'results/decision.json');assert decision['new_method_established'] is False
    assert decision['criteria']==study.decisions(formal,study.load(study.PARENTS['budget']/'curation/decision.json'))
    assert all(not r['old_failed_gate_relabelled'] and not r['new_method_established'] for r in decision['criteria'])
    spec=importlib.util.spec_from_file_location('clock20k_curator',root/'curation-source/curate.py')
    curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    labels={v:k for k,v in curator.LABELS.items()};reported=[]
    for line in (doc/'analysis-report.md').read_text().splitlines():
        cells=[c.strip().replace('**','') for c in line.strip().strip('|').split('|')]
        if len(cells)==4 and cells[0] in labels:reported.append(cells)
    assert len(reported)==9
    for label,*values in reported:
        original=next(r for r in formal if r['family']==labels[label])
        assert values==[format(original[k],p) for k,p in [('weighted_mse','.6f'),('clean_mse','.6f'),('fid','.2f')]]
    review=study.load(root/'visual-review.json');pngs={p.name for p in (root/'curation/figures').glob('*.png')}
    assert review['status']=='PASS' and len(pngs)==3 and set(review['files'])==pngs
    for name,digest in review['files'].items():assert study.sha(root/'curation/figures'/name)==digest
    assert len(list((doc/'results/figures').glob('*')))==6
    assert len(list(csv.DictReader((doc/'results/results.tsv').open(),delimiter='\t')))==9
    for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md','README.md'):assert (doc/name).exists()
    protected=['src','synthetic','configs','scripts','reports/report1']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
    links=0
    for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith(('/','mailto:')):continue
            assert (p.parent/unquote(target)).exists(),(p,target);links+=1
    for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    study.verify(root,True)
    study.write(doc/'results/artifact-verification.json',dict(status='PASS',new_interventions=2,reused_native20k=7,
        new_training=False,new_generated_images_replayed=2048,new_generated_features_replayed=2048,
        reference_features_replayed=2048,independent_FIDs=9,calibration_scalars=12,time_panel_scores=45,
        report_rows_verified=9,figures_visually_reviewed=3,local_links=links,protected_scopes=protected,new_method_established=False))
    shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
    for p in doc.iterdir():
        if p.is_file():shutil.copyfile(p,closure/p.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
    study.write(root/'sealed-runtime.json',{str(p.relative_to(root)):study.sha(p) for p in files})
    for name,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/name)==digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),
        private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
