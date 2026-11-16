"""NumPy-only reconstruction of every saved endpoint and original128 replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loss(x,y,d):return float(np.mean((np.sort(x@d.T,axis=0)-np.sort(y@d.T,axis=0))**2))


def main():
    p=argparse.ArgumentParser()
    for name in ('root','input','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    done=read(a.input/'COMPLETE.json');portable=read(a.root/'portable.json')
    assert done['passed'] and done['parameters_buffers_rng_unchanged']
    assert done['portable_sha256']==sha(a.root/'portable.json')
    rows=read(a.input/'rows.json');assert len(rows)==done['rows']
    maximum=0.;identities=set();replay_count=0
    for row in rows:
        key=(row['case'],row['split'],row['steps']);assert key not in identities;identities.add(key)
        case=next(c for c in portable['cases'] if c['id']==row['case'])
        files=case['splits'][row['split']];datafile=a.root/files['assets']['file']
        assert sha(datafile)==files['assets']['sha256']
        assert sha(a.root/case['tensor_file'])==case['tensor_sha256']
        with np.load(datafile) as z:
            d=z['directions'].astype(np.float64)
            targets=[z[f'target_{b}'][:row['points_per_bank']].astype(np.float64) for b in range(8)]
        file=a.input/row['file'];assert sha(file)==row['sha256']
        with np.load(file) as z:values={k:z[k].copy() for k in ('native','crossed')}
        for kind,x in values.items():
            assert x.shape==(8,row['points_per_bank'],targets[0].shape[1]) and np.isfinite(x).all()
            banks=[np.sqrt(loss(v,t,d)) for v,t in zip(x,targets,strict=True)]
            pool=np.sqrt(loss(np.concatenate(x),np.concatenate(targets),d))
            observed=np.array([row[kind]['pooled_sw2']]+row[kind]['bank_sw2'])
            expected=np.array([pool]+banks);error=np.abs(observed-expected)
            assert np.all(error<=1e-12+1e-10*np.abs(expected)),(key,kind,error)
            maximum=max(maximum,float(error.max()))
            if row['steps']==128:
                cachefile=a.root/files['cache_file'];assert sha(cachefile)==files['cache_sha256']
                with np.load(cachefile) as z:cached=np.stack([z[f'{kind}_{b}'][:row['points_per_bank']] for b in range(8)])
                assert np.all(np.abs(x-cached)<=1e-7*(1+np.abs(cached))),key
                expected_cached=np.array([np.sqrt(loss(np.concatenate(cached),np.concatenate(targets),d))]+
                    [np.sqrt(loss(v,t,d)) for v,t in zip(cached,targets,strict=True)])
                assert np.all(np.abs(expected-expected_cached)<=2e-8+1e-7*np.abs(expected_cached))
                assert row['replay'][kind]['qualified'];replay_count+=1
        print('VERIFIED_RESOLUTION',*key,flush=True)
    cases={(r['case'],r['split']) for r in rows}
    expected_steps={128} if done['witness'] else {127,128,129,512,2048}
    assert all({r['steps'] for r in rows if (r['case'],r['split'])==case}==expected_steps for case in cases)
    assert all(r['points_per_bank']==(64 if done['witness'] else 4096) for r in rows)
    result=dict(passed=True,witness=done['witness'],cases=len(cases),rows=len(rows),
        independently_reconstructed_metrics=len(rows)*18,max_metric_error=maximum,
        original128endpoint_and_metric_replays=replay_count,portable_sha256=done['portable_sha256'],
        complete_sha256=sha(a.input/'COMPLETE.json'),rows_sha256=sha(a.input/'rows.json'),
        source_sha256=sha(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('NUMPY_POPULATION_RESOLUTION_PASS',len(rows),maximum,flush=True)


if __name__=='__main__':main()
