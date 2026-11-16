"""Reconstruct copied-evaluator field, sample and case metrics in NumPy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from verify_rk4_samples import sha, sw2


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', required=True)
    args = parser.parse_args()
    root,out,precision = map(Path,[args.input,args.output,args.precision])
    out.mkdir(parents=True,exist_ok=False)
    complete,binding = read(root/'COMPLETE.json'),read(root/'binding.json')
    assert complete['passed'] and complete['steps']==[20000]
    anchor = read(root/'native-anchor.json')
    assert anchor['passed'] and anchor['exact']
    folder = Path(binding['source_folder'])
    assert anchor['checkpoint_sha256']==sha(folder/'checkpoints/step_0020000.pt')
    asset = binding['assets']
    assert sha(Path(asset['file']))==asset['sha256']
    with np.load(asset['file']) as z:
        arrays = {key:z[key].copy() for key in z.files}
    summary = read(root/'summary.json')[0]
    assert summary['step']==20000 and summary['inference_solver']==binding['inference_solver']
    assert summary['checkpoint_sha256']==anchor['checkpoint_sha256']
    field = np.load(root/'field-20000.npy')
    mse = float(np.mean((field.astype(float)-arrays['field_oracle'].astype(float))**2))
    cell = binding['source_config']['cell']
    reference = read(precision/'PASS.json')
    assert reference['passed'] and reference['split']==binding['split']
    assert reference['source_binding_sha256']==sha(root/'binding.json')
    assert reference['asset_sha256']==asset['sha256']
    assert reference['checkpoint_sha256']==anchor['checkpoint_sha256']
    assert reference['inference_solver']==binding['inference_solver']
    assert reference['field_x_sha256']==hashlib.sha256(arrays['field_x'].tobytes()).hexdigest()
    assert reference['field_t_sha256']==hashlib.sha256(arrays['field_t'].tobytes()).hexdigest()
    assert reference['prediction_sha256']==sha(precision/'field-fp64.npy')
    double_field = np.load(precision/'field-fp64.npy')
    assert np.allclose(field,double_field,atol=3e-5,rtol=3e-5)
    banks = 1 if complete['witness'] else 8
    n = 64 if complete['witness'] else 4096
    samples,targets,scores,hashes = [],[],[],[]
    for bank in range(banks):
        file = root/f'20000-b{bank}.npy'
        value,target = np.load(file),arrays[f'target_{bank}'][:n]
        assert value.shape==target.shape and len(value)==n and np.isfinite(value).all()
        samples.append(value)
        targets.append(target)
        scores.append(sw2(value,target,arrays['directions'].astype(float)))
        hashes.append(dict(file=str(file),sha256=sha(file)))
    samples,targets = np.concatenate(samples),np.concatenate(targets)
    metrics = dict(field_mse=mse,bank_sw2=scores,sw2_mean=float(np.mean(scores)),
        sw2_sd=float(np.std(scores,ddof=1)) if banks>1 else 0.,
        pooled_sw2=sw2(samples,targets,arrays['directions'].astype(float)))
    manifest = read(Path(binding['source_config']['data_root'])/'manifest.json')
    law = manifest['datasets'][cell['dataset']]['distribution']
    if samples.shape[1]==2:
        counts = np.bincount(np.argmax(samples@np.array(law['means']).T,axis=1),minlength=8)
        metrics.update(counts=counts.tolist(),sector_l1=float(np.abs(counts/len(samples)-law['weights']).sum()))
    else:
        latent = samples.astype(float)@np.asarray(law['rotation'])
        observed,expected = [],[]
        for axis in [0,7,15]:
            location,scale = law['locations'][axis],math.sqrt(law['variances'][axis])
            phi = lambda z:(1+math.erf(z/math.sqrt(2)))/2
            expected.append(phi(-location/(2*scale))-phi(-3*location/(2*scale)))
            observed.append(float(np.mean(np.abs(latent[:,axis])<location/2)))
        metrics.update(gap_axes=[0,7,15],gap_mass=observed,gap_expected=expected,
                       gap_error_max=float(np.max(np.abs(np.asarray(observed)-expected))))
    maximum = max(float(np.max(np.abs(np.asarray(v)-np.asarray(summary[k])))) for k,v in metrics.items())
    assert maximum<1e-10
    result = dict(passed=True,witness=complete['witness'],dataset=cell['dataset'],id=cell['id'],
        inference_solver=binding['inference_solver'],split=binding['split'],metrics=metrics,
        checkpoint_sha256=anchor['checkpoint_sha256'],asset_sha256=asset['sha256'],sample_files=hashes,
        maximum_independent_error=maximum,field_precision_conversion_max_error=float(np.max(np.abs(field-double_field))),
        input_complete_sha256=sha(root/'COMPLETE.json'),precision_sha256=sha(precision/'PASS.json'),
        source_sha256=sha(Path(__file__)))
    (out/'PASS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,id=cell['id'],split=binding['split'],error=maximum)),flush=True)


if __name__=='__main__':
    main()
