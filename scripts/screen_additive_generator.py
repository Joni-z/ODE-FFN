"""Apply the preregistered additive-generator practical screen to audited data."""
import argparse
import hashlib
import json
from pathlib import Path

NONLINEAR=('gmm_ring8_2d','rotated_product_bimodal_16d')


def screen(studies,output):
    rows=[];inputs={}
    for path in sorted(studies.glob('*/audit/audit.json')):
        audit=json.loads(path.read_text())
        assert audit['state']=='PASS' and audit['source_verified'] and audit['oracles_recomputed']
        assert audit['common_inputs_match']
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();rows.extend(audit['rows'])
    assert len(rows)==18
    datasets=sorted({r['dataset'] for r in rows})
    assert set(datasets)=={'gaussian_aniso_2d',*NONLINEAR}
    eligible=[];comparisons={}
    for family in ('scalar','vector'):
        arm,parent=family+'_add',family+'_mul';measured={}
        for dataset in datasets:
            data={r['label']:r for r in rows if r['dataset']==dataset}
            candidate,reference=data[arm],data[parent];baselines=[data['swiglu_budget'],data['swiglu']]
            assert candidate['parameters']==reference['parameters']
            assert candidate['forward_macs']==reference['forward_macs']
            measured[dataset]=dict(oracle_mse=candidate['validation_oracle_mse'],forward_macs=candidate['forward_macs'],
                oracle_over_parent=candidate['validation_oracle_mse']/reference['validation_oracle_mse'],
                oracle_over_better_swiglu=candidate['validation_oracle_mse']/min(b['validation_oracle_mse'] for b in baselines),
                jacobian_over_better_swiglu=candidate['jacobian_error']/min(b['jacobian_error'] for b in baselines),
                sw2_over_better_swiglu=candidate['validation_sw2']/min(b['validation_sw2'] for b in baselines),
                wall_over_parent=candidate['wall_seconds']/reference['wall_seconds'])
        leads=[d for d in NONLINEAR if measured[d]['oracle_over_better_swiglu']<=.9 and measured[d]['oracle_over_parent']<1]
        passes=bool(leads) and all(m['oracle_over_better_swiglu']<=1.05 for m in measured.values()) and all(
            measured[d]['jacobian_over_better_swiglu']<=1.1 and measured[d]['sw2_over_better_swiglu']<=1.1 for d in NONLINEAR)
        comparisons[arm]=dict(parent=parent,leads=leads,passes=passes,datasets=measured)
        if passes:eligible.append(arm)
    eligible.sort(key=lambda arm:(comparisons[arm]['datasets'][NONLINEAR[1]]['oracle_mse'],
        comparisons[arm]['datasets'][NONLINEAR[1]]['forward_macs'],comparisons[arm]['datasets'][NONLINEAR[0]]['oracle_mse']))
    result=dict(state='CONTINUE' if eligible else 'STOP',selected=eligible[0] if eligible else None,
        eligible=eligible,comparisons=comparisons,audit_inputs_sha256=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Registered practical engineering screen; no inference, fresh-holdout claim or full architecture completion.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=screen(args.studies,args.output)
    print(json.dumps(dict(state=result['state'],eligible=result['eligible'])))
