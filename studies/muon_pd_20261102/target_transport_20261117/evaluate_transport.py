"""Frozen-weight learned-field tangent, target derivative and actual interventions."""
import argparse
import copy
import os
from pathlib import Path
import time

import numpy as np
from common import base, ev, assets, torch, read, dump, restored_pair
from learned_transport import (LAMBDAS, BlendedField, SensitivityField, integrate,
                               prediction_summary, squared_sw2)


def field_jvp_check(model, arrays: dict) -> dict:
    x, t = (torch.tensor(arrays['field_'+name][:64],dtype=torch.float64) for name in ('x','t'))
    direction = torch.cos(torch.arange(x.numel(),dtype=x.dtype)).reshape_as(x)
    with torch.no_grad():
        _, actual = torch.func.jvp(lambda value:model(value,t),(x,),(direction,))
        epsilon = 1e-5
        expected = (model(x+epsilon*direction,t)-model(x-epsilon*direction,t))/(2*epsilon)
    error = (actual-expected).abs()
    ok = error <= 2e-6+2e-5*expected.abs()
    return dict(points=len(x),direction='cos(arange(x.numel()))',epsilon=epsilon,
        max_error=float(error.max()),failed_components=int((~ok).sum()),qualified=bool(ok.all()))


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('cases','case','data-root','assets','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--split',choices=('selection','confirmation'),required=True)
    parser.add_argument('--witness',action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    out = Path(args.output);out.mkdir(parents=True,exist_ok=False)
    before = time.monotonic()
    case = next(c for c in read(args.cases)['cases'] if c['id'] == args.case)
    base.verify_sources()
    arrays, _, binding = assets(args.assets,case['dataset'],args.split,'cpu')
    model32, crossed32, identity = restored_pair(case,args.data_root)
    native_snapshot, crossed_snapshot = ev.tensor_snapshot(model32), ev.tensor_snapshot(crossed32)
    # Conversion preserves the actual restored FP32 values of nonpersistent buffers.
    model64, crossed64 = copy.deepcopy(model32).double(), copy.deepcopy(crossed32).double()
    snapshot64, crossed_snapshot64 = ev.tensor_snapshot(model64), ev.tensor_snapshot(crossed64)
    jvp = field_jvp_check(model64,arrays)
    dump(out/'binding.json',dict(case=case,split=args.split,assets=binding,
        assets_manifest_sha256=base.digest(Path(args.assets)/'manifest.json'),
        cases_sha256=base.digest(args.cases),source_sha256=base.digest(__file__),
        heun_steps=128,lambdas=list(LAMBDAS),training_seed=0,witness=args.witness,**identity))
    dump(out/'field-jvp.json',jvp)
    count, rows = (1 if args.witness else 8), []
    pooled = dict(native=[],tangent=[],endpoints=[],native32=[],crossed32=[],targets=[])
    directions = arrays['directions'].astype(np.float64)

    def progress(bank: int, stage: str) -> None:
        dump(out/'progress.json',dict(pid=os.getpid(),case=args.case,split=args.split,
            bank=bank,banks=count,stage=stage,completed_banks=len(rows),seconds=time.monotonic()-before))
        print(args.case,args.split,bank,stage,flush=True)

    for bank in range(count):
        source_array = arrays[f'source_{bank}'][:64] if args.witness else arrays[f'source_{bank}']
        target = arrays[f'target_{bank}'][:64] if args.witness else arrays[f'target_{bank}']
        target = target.astype(np.float64)
        source32 = torch.from_numpy(source_array)
        source64 = source32.double()
        progress(bank,'native-fp32')
        native32 = base.train.generated_samples(model32,source32,128).numpy()
        alternate32 = base.train.generated_samples(crossed32,source32,128).numpy()
        progress(bank,'native-fp64')
        native64, native_path = integrate(base.train.generated_samples,model64,source64)
        alternate64, alternate_path = integrate(base.train.generated_samples,crossed64,source64)
        progress(bank,'learned-tangent')
        sensitivity = SensitivityField(model64,crossed64)
        initial = torch.cat((source64,torch.zeros_like(source64)),dim=1)
        augmented, augmented_path = integrate(base.train.generated_samples,sensitivity,initial)
        dim = source64.shape[1]
        replay_exact = torch.equal(native64,augmented[:,:dim])
        replay_path_exact = torch.equal(native_path,augmented_path[:,:,:dim])
        tangent = augmented[:,dim:].numpy()
        endpoints, paths = [], []
        for coefficient in LAMBDAS:
            progress(bank,'lambda-'+str(coefficient))
            if coefficient == 1:
                endpoint, path = alternate64, alternate_path
            else:
                endpoint, path = integrate(base.train.generated_samples,
                    BlendedField(model64,crossed64,coefficient),source64)
            endpoints.append(endpoint.numpy());paths.append(path.numpy())
        endpoints = np.stack(endpoints)
        progress(bank,'endpoint-finite-difference')
        epsilon = 1e-4
        prefix = source64[:64]
        plus = base.train.generated_samples(BlendedField(model64,crossed64,epsilon),prefix,128)
        minus = base.train.generated_samples(BlendedField(model64,crossed64,-epsilon),prefix,128)
        finite_difference = ((plus-minus)/(2*epsilon)).numpy()
        derivative_error = np.abs(finite_difference-tangent[:64])
        derivative_ok = derivative_error <= 2e-5+1e-3*np.abs(finite_difference)
        fd = dict(points=len(prefix),epsilon=epsilon,max_error=float(derivative_error.max()),
                  failed_components=int((~derivative_ok).sum()),qualified=bool(derivative_ok.all()))
        numeric = dict(primal_exact=replay_exact,path_exact=replay_path_exact,
            jvp_primal_error=sensitivity.primal_max_error,jvp_primal_qualified=sensitivity.primal_qualified,
            endpoint_finite_difference=fd)
        native = native64.numpy()
        summary = prediction_summary(native,tangent,endpoints,target,directions)
        loss32 = squared_sw2(native32.astype(float),target,directions)
        change32 = squared_sw2(alternate32.astype(float),target,directions)-loss32
        change64 = summary['rows'][-1]['actual_loss_change']
        precision = dict(native_endpoint_rms=float(np.sqrt(np.mean((native32-native)**2))),
            crossed_endpoint_rms=float(np.sqrt(np.mean((alternate32-endpoints[-1])**2))),
            native_endpoint_max=float(np.max(np.abs(native32-native))),
            crossed_endpoint_max=float(np.max(np.abs(alternate32-endpoints[-1]))),
            native_sw2_fp32=float(np.sqrt(loss32)),native_sw2_fp64=summary['derivative']['sw2'],
            crossed_sw2_fp32=float(np.sqrt(loss32+change32)),crossed_sw2_fp64=summary['rows'][-1]['sw2'],
            loss_change_fp32=change32,loss_change_fp64=change64,
            loss_change_precision_error=abs(change32-change64),
            direction_robust=bool(np.sign(change32)==np.sign(change64) and abs(change32-change64)<abs(change64)))
        file = out/f'bank{bank}.npz'
        np.savez_compressed(file,native32=native32,crossed32=alternate32,native=native,
            tangent=tangent,endpoints=endpoints,native_path=native_path.numpy(),
            augmented_native=augmented[:,:dim].numpy(),
            augmented_native_path=augmented_path[:,:,:dim].numpy(),
            tangent_path=augmented_path[:,:,dim:].numpy(),paths=np.stack(paths),
            finite_difference=finite_difference)
        numeric['qualified'] = bool(jvp['qualified'] and replay_exact and replay_path_exact and
            sensitivity.primal_qualified and fd['qualified'] and summary['derivative']['qualified'])
        rows.append(dict(bank=bank,points=len(native),file=file.name,sha256=base.digest(file),
                         numerical=numeric,precision=precision,**summary))
        for name,value in [('native',native),('tangent',tangent),('endpoints',endpoints),
                           ('native32',native32),('crossed32',alternate32),('targets',target)]:
            pooled[name].append(value)
        dump(out/'banks.json',rows)
        progress(bank,'bank-complete')
    x, u, target = (np.concatenate(pooled[name]) for name in ('native','tangent','targets'))
    endpoints = np.concatenate(pooled['endpoints'],axis=1)
    pool = prediction_summary(x,u,endpoints,target,directions)
    x32, alt32 = (np.concatenate(pooled[name]).astype(float) for name in ('native32','crossed32'))
    loss32, altloss32 = (squared_sw2(values,target,directions) for values in (x32,alt32))
    pool['native_fp32_sw2'], pool['crossed_fp32_sw2'] = np.sqrt(loss32), np.sqrt(altloss32)
    pool['fp32_loss_change'] = altloss32-loss32
    pool['direction_robust_to_precision'] = bool(np.sign(altloss32-loss32)==np.sign(pool['rows'][-1]['actual_loss_change']) and
        abs(altloss32-loss32-pool['rows'][-1]['actual_loss_change'])<abs(pool['rows'][-1]['actual_loss_change']))
    for model,snapshot in [(model32,native_snapshot),(crossed32,crossed_snapshot),
                           (model64,snapshot64),(crossed64,crossed_snapshot64)]:
        ev.same_tensors(model,snapshot)
    qualified = all(row['numerical']['qualified'] for row in rows) and pool['derivative']['qualified']
    result = dict(case=args.case,dataset=case['dataset'],split=args.split,witness=args.witness,
        banks=rows,pooled=pool,all_numerically_qualified=qualified,
        field_jvp=jvp,seconds=time.monotonic()-before)
    dump(out/'summary.json',result)
    dump(out/'COMPLETE.json',dict(passed=True,all_numerically_qualified=qualified,
        banks=count,samples_per_bank=len(source_array),parameters_and_buffers_unchanged=True,
        witness=args.witness,summary_sha256=base.digest(out/'summary.json')))
    print('TARGET_TRANSPORT_COMPLETE',args.case,args.split,qualified,flush=True)


if __name__ == '__main__':main()
