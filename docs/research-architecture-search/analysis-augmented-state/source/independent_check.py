"""Read-only Stage24 raw-metric and registered-screen cross-check, after terminal audit."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import psutil

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/mr.z/ODE-FFN-report-work')
OUT=REPO/'docs/research-architecture-search/analysis-augmented-state'
DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
GENERIC=('swiglu','silu128','swiglu_budget','sigmoid_glu')
METRICS=('validation_oracle_mse','jacobian_error','validation_sw2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    hashes={}
    def read(path):
        hashes[str(path)]=sha(path)
        return json.loads(Path(path).read_text())
    assert read(ROOT/'output/augmented_state_queue_20261025_m1/status.json')['state']=='COMPLETE'
    assert read(OUT/'summary.json')['state']=='PASS'
    rows=list(csv.DictReader((OUT/'comparison.csv').open()))
    hashes[str(OUT/'comparison.csv')]=sha(OUT/'comparison.csv')
    assert len(rows)==24
    lookup={(r['dataset'],r['label']):r for r in rows}
    def value(dataset,label,metric):return float(lookup[dataset,label][metric])
    raw_checks=[]
    for suffix in ('v1','product16'):
        study=ROOT/'output'/f'augmented_state_{suffix}_20261025_m1'
        assert read(study/'status.json')['state']=='COMPLETE'
        raw=read(study/'results.json')
        historical=ROOT.parent/'affine_forcing_20261025_m1/output'/f'affine_forcing_{suffix}_20261025_m1'
        old=[r for r in read(historical/'results.json') if r['label'] in ('swiglu_budget','sigmoid_glu')]
        for record in raw+old:
            selected=lookup[record['dataset'],record['label']]
            for key in (*METRICS,'field_mse','parameters','forward_macs'):
                assert float(selected[key])==float(record[key]),(record['label'],key)
            raw_checks.append(dict(dataset=record['dataset'],label=record['label']))
    assert len(raw_checks)==24
    decisions={};frontier={}
    for dataset in DATASETS:
        frontier[dataset]={metric:min((value(dataset,label,metric),label) for label in GENERIC)
                           for metric in METRICS}
    for candidate in ('lowrank_budget','lowrank_narrow','lowrank_augmented'):
        ratios={d:{m:value(d,candidate,m)/frontier[d][m][0] for m in METRICS} for d in DATASETS}
        primary=all(ratios[d][METRICS[0]]<=1.05 for d in DATASETS)
        secondary=all(ratios[d][m]<=1.1 for d in DATASETS[1:] for m in METRICS[1:])
        qualifying=[d for d in DATASETS[1:] if ratios[d][METRICS[0]]<=.9 and
                    value(d,candidate,METRICS[0])/value(d,'vector_parent',METRICS[0])<=.9]
        saving=all(value(d,candidate,m)/value(d,'swiglu',m)<=.8
                   for d in DATASETS for m in ('parameters','forward_macs'))
        practical=primary and secondary and (saving if candidate=='lowrank_narrow' else bool(qualifying))
        decisions[candidate]=dict(practical_lead=practical,all_primary_within_5pct=primary,
            nonlinear_Jacobian_SW2_within_10pct=secondary,qualifying_nonlinear_tasks=qualifying,
            at_least_20pct_full_model_cost_savings=saving,ratios=ratios)
    allocation_tasks=[d for d in decisions['lowrank_augmented']['qualifying_nonlinear_tasks'] if all(
        value(d,'lowrank_augmented',METRICS[0])/value(d,c,METRICS[0])<=.9
        for c in ('lowrank_budget','lowrank_narrow'))]
    allocation=decisions['lowrank_augmented']['practical_lead'] and bool(allocation_tasks) and all(
        value(d,'lowrank_augmented',METRICS[0])/value(d,c,METRICS[0])<=1.05
        for d in DATASETS for c in ('lowrank_budget','lowrank_narrow'))
    formal=read(OUT/'screen-decision.json')
    for label,result in decisions.items():
        for key,val in result.items():
            if key!='ratios':assert formal['decisions'][label][key]==val
    leads=[a for a,r in decisions.items() if r['practical_lead']]
    assert formal['practical_leads']==leads and formal['augmentation_allocation_evidence']==allocation
    assert formal['augmentation_qualifying_tasks']==allocation_tasks and not formal['architecture_goal_complete']
    source_counts={}
    for root in (ROOT,ROOT.parent/'affine_forcing_20261025_m1',ROOT.parent/'diagonal_flow_20261024_m1'):
        manifest=read(root/'source-manifest.json');source=root/'source'
        paths={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        assert paths==set(manifest['all_files'])
        for path,digest in manifest['all_files'].items():assert sha(source/path)==digest
        source_counts[root.name]=len(paths)
    assert source_counts[ROOT.name]==604 and source_counts['affine_forcing_20261025_m1']==595
    provenance=read(OUT/'provenance.json')
    for path,digest in provenance['input_sha256'].items():assert sha(path)==digest
    live=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            command=' '.join(process.info['cmdline'] or [])
            if str(ROOT/'source/scripts/run_architecture_study.py') in command or (
                str(ROOT/'output') in command and 'synthetic.train' in command):live.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not live,live
    receipt=dict(state='PASS',source_sha256=sha(__file__),raw_result_rows=24,frontier=frontier,
        independently_recomputed_decisions=decisions,practical_leads=leads,
        augmentation_allocation_evidence=allocation,source_files=source_counts,
        analysis_input_bindings_verified=len(provenance['input_sha256']),live_formal_processes=live,
        input_sha256=hashes,architecture_goal_complete=False,
        scope='Independent raw-table arithmetic and fixed decision guards. Native CP/array replays are separately executed by the main analyzer, not duplicated here.')
    (OUT/'independent-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    shutil.copy2(__file__,OUT/'source/independent_check.py')
    print(json.dumps({'state':'PASS','raw_rows':24,'leads':leads,'allocation':allocation,'input_hashes':receipt['analysis_input_bindings_verified']}))


if __name__=='__main__':main()
