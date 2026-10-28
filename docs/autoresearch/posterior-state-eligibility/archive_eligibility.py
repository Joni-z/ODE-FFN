"""Archive a data-free identity check without promoting it to an ML result."""
import argparse,ast,csv,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2];assert not (root/'sealed-runtime.json').exists()
    bindings=load(root/'bindings.json')
    for path,digest in bindings.items():assert sha(path)==digest
    assert sha(doc/'check_identities.py')==sha(root/'source/check_identities.py')
    assert sha(doc/'inspect_arithmetic.py')==sha(root/'postprocessing-source/inspect_arithmetic.py')
    for name,session,chunk in [('tool-exit.json',2293,'e0631f'),('arithmetic-tool-exit.json',41418,'224125')]:
        receipt=load(root/name);assert receipt['exec_session']==session and receipt['exit_code']==0
        assert receipt['terminal_tool_chunk']==chunk
        assert load(doc/'results'/name)==receipt
    for path in (root/'results').iterdir():assert sha(path)==sha(doc/'results'/path.name)
    assert sha(root/'tolerance-witness.json')==sha(doc/'results/tolerance-witness.json')
    assert sha(root/'results.tsv')==sha(doc/'results/results.tsv')
    norm=load(root/'norm-input-source-check.json');assert norm==load(doc/'results/norm-input-source-check.json')
    assert norm['status']=='PASS' and len(norm['files'])==2
    for row in norm['files']:assert sha(Path(row['frozen_native_path']))==sha(repo/row['current_repo_path'])==row['sha256']
    summary=load(root/'results/summary.json')
    assert summary['status']=='PASS' and summary['data_free'] and summary['fixed_priors']==6
    assert summary['source_commit']==load(root/'settings.json')['source_commit']
    assert not any(summary[k] for k in ('neural_training','image_prior_fitted','new_quality_score','solver_repackaging_nominated'))
    assert summary['environment']['device']=='cpu' and summary['environment']['threads']==1 and summary['seconds']<120
    read=lambda p:list(csv.DictReader(p.open()))
    fixed=read(root/'results/fixed-observation.csv');times=read(root/'results/FM-coordinates.csv');integrated=read(root/'results/integrated-states.csv')
    assert (len(fixed),len(times),len(integrated))==(72,72,54)
    families={v['name'] for v in load(root/'results/priors.json')};assert len(families)==6
    assert {(r['prior'],int(r['observation']),float(r['precision'])) for r in fixed}=={(n,i,l) for n in families for i in range(3) for l in (0,.2,1,4)}
    assert {(r['prior'],int(r['state']),float(r['time'])) for r in times}=={(n,i,t) for n in families for i in range(3) for t in (.15,.5,.85,.98)}
    assert {(r['prior'],int(r['observation']),float(r['precision'])) for r in integrated}=={(n,i,l) for n in families for i in range(3) for l in (.2,1,4)}
    for r in fixed:
        assert max(float(r[k]) for k in ('conditioning_max','cubature_max','precision_state_max','ordinary_endpoint_max'))<3e-10
    for r in times:
        assert max(float(r[k]) for k in ('fixed_z_max','state_Jacobian_max','material_max','ordinary_endpoint_max'))<3e-10
        assert (r['cap_active']=='True')==(float(r['time'])==.98)
    assert max(float(r['full_state_and_mean_max']) for r in integrated)==summary['integrated_state_maximum']<5e-8
    assert min(float(r['minimum_covariance_eigenvalue']) for r in integrated)>-3e-10
    assert max(float(r['logit_ratio_Hessian_norm']) for r in fixed if r['prior']=='unequal3')==summary['unequal_covariance_logit_ratio_Hessian_max']>1e-2
    assert max(float(r['dropped_observation_term_error']) for r in times)==summary['dropping_FM_observation_term_max']>1e-2
    witness=load(root/'tolerance-witness.json');actual=np.array(witness['actual_vector']);expected=np.array(witness['expected_vector'])
    bounds=3e-10+3e-10*np.abs(expected);errors=np.abs(actual-expected)
    assert np.all(errors<=bounds) and not witness['tolerances_changed']
    assert errors.max()==max(float(r['complete_component_time_max']) for r in times)==summary['maximum_absolute_errors']['FM_complete_component_time']
    np.testing.assert_allclose((errors/bounds).max(),witness['maximum_fraction_of_allowed_error'],rtol=0,atol=0)
    counter=load(root/'results/counterexample.json');assert len(counter)==2
    for row in counter:
        np.testing.assert_allclose(row['mean'],[0],rtol=0,atol=1e-15)
        np.testing.assert_allclose(row['covariance'],[[1]],rtol=0,atol=1e-15)
    np.testing.assert_allclose(counter[1]['third'],[1.5],rtol=0,atol=1e-15)
    np.testing.assert_allclose(counter[1]['correct_precision_drift'],[-.75],rtol=0,atol=1e-15)
    decision=load(doc/'results/decision.json')
    assert decision['status']=='DO_NOT_TRAIN_SOLVER_REPACKAGING' and decision['exact_ordinary_endpoint']
    assert not decision['new_ODELayer_candidate'] and not decision['Gaussian_image_reference_reopened']
    assert decision['does_not_rule_out_useful_learned_parameter_constraints']
    ledger=list(csv.DictReader((doc/'results/results.tsv').open(),delimiter='\t'))
    assert len(ledger)==1 and ledger[0]['metric']=='NA' and ledger[0]['status']=='discard'
    protected=['src','synthetic','configs','scripts','reports/report1']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
    tracked=set(subprocess.check_output(['git','-C',str(repo),'ls-files','-z'],text=True).split('\0'));links=0
    for path in doc.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            target=target.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith('/'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists() and str(resolved.relative_to(repo)) in tracked,(path,target)
            links+=1
    for path in doc.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for path,digest in bindings.items():assert sha(path)==digest
    write(doc/'results/artifact-verification.json',dict(status='PASS',fixed_observation_cases=72,FM_coordinate_cases=72,integrated_endpoints=54,
        original_tool_exits_observed=True,nonzero_unequal_covariance_Hessian=True,FM_observation_term_retained=True,
        declared_relative_and_absolute_tolerances_unchanged=True,raw_counterexample_retained=True,parent_seal_manifests_unchanged=True,
        full_parent_raw_file_rehash_performed=False,protected_scopes=protected,local_links=links,new_model_or_quality_result=False))
    shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
    for path in doc.iterdir():
        if path.is_file():shutil.copyfile(path,closure/path.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
    write(root/'sealed-runtime.json',{str(p.relative_to(root)):sha(p) for p in files})
    for name,digest in load(root/'sealed-runtime.json').items():assert sha(root/name)==digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=sha(root/'sealed-runtime.json'),private_bytes=sum(p.stat().st_size for p in files),new_quality_result=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root.resolve())
