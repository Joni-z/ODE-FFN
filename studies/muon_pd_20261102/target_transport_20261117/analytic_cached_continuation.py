"""Reuse every saved lambda to separate field approximation and finite response."""
import argparse
from pathlib import Path
import numpy as np
from population_resolution import read, dump, sha
from verify_analytic_target import reconstruct

LAMBDAS=(.125,.25,.5,1.)


def main():
    p=argparse.ArgumentParser()
    for name in ('diagnostic','main-root','old-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    groups=[]
    for current in read(a.diagnostic)['rows']:
        assert current['steps']==128 and current['banks']==8 and current['points']==32768
        case,split=current['case'],current['split']
        old=a.old_root/split/case
        summary=read(old/'summary.json');complete=read(old/'COMPLETE.json');binding=read(old/'binding.json')
        assert complete['passed'] and not complete['witness'] and complete['summary_sha256']==sha(old/'summary.json')
        assert [r['coefficient'] for r in summary['pooled']['rows']]==list(LAMBDAS)
        asset=Path(binding['assets']['file']);assert sha(asset)==binding['assets']['sha256']
        endpoints=[];targets=[];replays=[];source_hashes=[]
        with np.load(asset) as original_assets:
            directions=original_assets['directions'].astype(np.float64)
            for bank,row in enumerate(summary['banks']):
                assert row['bank']==bank and row['points']==4096
                file=old/row['file'];assert sha(file)==row['sha256']
                modern=a.main_root/f'{case}-{split}-bank{bank}-heun128'
                new=read(modern/'COMPLETE.json');assert new['passed'] and new['bank']==bank
                expected=current['bank_bindings'][bank]
                assert expected['complete_sha256']==sha(modern/'COMPLETE.json')
                assert expected['array_sha256']==sha(modern/new['file'])
                with np.load(file) as data,np.load(modern/new['file']) as z:
                    assert np.array_equal(z['directions'],directions)
                    assert np.array_equal(z['target'],original_assets['target_'+str(bank)].astype(np.float64))
                    assert data['endpoints'].shape==(4,4096,z['native'].shape[1])
                    errors=[]
                    for left,right in [(data['native'],z['native_reference']),
                                       (data['endpoints'][-1],z['crossed_reference'])]:
                        error=np.abs(left-right)
                        assert np.all(error<=1e-10+1e-9*np.abs(right))
                        errors.append(float(error.max()))
                    replays.append(dict(bank=bank,native_max_error=errors[0],crossed_max_error=errors[1]))
                    endpoints.append(data['endpoints'].copy());targets.append(z['target'].copy())
                source_hashes.append(dict(bank=bank,original_array_sha256=sha(file),main_array_sha256=new['sha256']))
        target=np.concatenate(targets);pooled=np.concatenate(endpoints,axis=1)
        loss0=current['native_sw2']**2
        assert abs(np.sqrt(loss0)-summary['pooled']['derivative']['sw2'])<=1e-10
        la=current['predictions']['analytic']['predicted_squared_loss_change']
        lm=current['predictions']['measured']['predicted_squared_loss_change']
        assert abs(lm-summary['pooled']['derivative']['loss_derivative'])<=1e-10
        rows=[]
        for index,coefficient in enumerate(LAMBDAS):
            loss,_=reconstruct(pooled[index],target,directions)
            recorded=summary['pooled']['rows'][index]
            assert abs(loss-recorded['loss'])<=1e-12
            actual=loss-loss0
            assert abs(actual-recorded['actual_loss_change'])<=1e-12
            if coefficient==1:
                assert abs(actual-current['actual_squared_loss_change'])<=1e-12
            forecasts={}
            for name,derivative in [('analytic',la),('measured',lm)]:
                predicted=coefficient*derivative
                forecasts[name]=dict(predicted_loss_change=predicted,
                    score=None if actual==0 else 1-(predicted-actual)**2/actual**2,
                    sign_agrees=bool(np.sign(predicted)==np.sign(actual)))
            field=coefficient*(la-lm);finite=coefficient*lm-actual
            total=coefficient*la-actual
            assert abs(field+finite-total)<=1e-15
            rows.append(dict(coefficient=coefficient,actual_loss_change=actual,forecasts=forecasts,
                error_decomposition=dict(analytic_field_approximation=field,finite_intervention=finite,total=total)))
        groups.append(dict(case=case,split=split,rows=rows,replays=replays,source_hashes=source_hashes,
            old_summary_sha256=sha(old/'summary.json'),asset_sha256=sha(asset),
            old_all_numerically_qualified=complete['all_numerically_qualified'],
            main_prefix_qualified=current['all_main_prefix_checks_qualify']))
    dump(a.output,dict(passed=True,scope='retrospective cached continuation; all original gates unchanged',
        groups=groups,diagnostic_sha256=sha(a.diagnostic),source_sha256=sha(__file__),
        registered_primary_2048_unchanged=True,new_model_calls=0,new_samples=0,fitted_coefficients=0))
    print('CACHED_ANALYTIC_CONTINUATION_PASS',len(groups),flush=True)


if __name__=='__main__':main()
