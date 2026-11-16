"""Independently evaluate every secondary metric using the native Torch SW2."""
import argparse
import json
from pathlib import Path

import numpy as np
from metric_remainder import read,sha,decompose
from common import base,torch,dump


def native_loss(x: np.ndarray,target: np.ndarray,directions: np.ndarray) -> float:
    tensors=[torch.from_numpy(np.ascontiguousarray(value)).double() for value in (x,target,directions)]
    return base.train.sliced_w2(*tensors)**2


def check_rows(x,u,endpoints,target,directions,rows,derivative) -> list[float]:
    initial=native_loss(x,target,directions);errors=[]
    for actual,row in zip(endpoints,rows,strict=True):
        coefficient=row['coefficient']
        predicted=native_loss(x+coefficient*u,target,directions)
        actual_loss=native_loss(actual,target,directions)
        expected=dict(baseline_loss=initial,actual_loss=actual_loss,linear_endpoint_loss=predicted,
            actual_loss_change=actual_loss-initial,linear_loss_change=coefficient*derivative,
            endpoint_metric_change=predicted-initial,transport_remainder=actual_loss-predicted,
            functional_remainder=predicted-initial-coefficient*derivative,
            total_remainder=actual_loss-initial-coefficient*derivative)
        for key,value in expected.items():errors.append(abs(value-row[key]))
        assert abs(row['total_remainder']-row['transport_remainder']-row['functional_remainder'])<1e-12
    return errors


def controls() -> dict:
    records=[]
    for name,x,u,target,curve in [
        ('translation',np.array([[-2.],[0.],[1.]]),np.ones((3,1))*.2,np.array([[-1.8],[.2],[1.2]]),np.zeros((3,1))),
        ('rank_crossing',np.array([[-2.],[0.],[1.]]),np.array([[3.],[-2.],[.5]]),np.array([[-.3],[.4],[1.1]]),np.array([[.2],[-.1],[.3]])),
        ('zero',np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1)))]:
        directions=np.ones((1,1))
        xt=torch.tensor(x,requires_grad=True)
        residual=torch.sort(xt,dim=0).values-torch.sort(torch.tensor(target),dim=0).values
        gradient,=torch.autograd.grad(residual.square().mean(),xt)
        derivative=float((gradient*torch.tensor(u)).sum())
        endpoints=np.stack([x+c*u+c*c*curve for c in (.125,.25,.5,1.)])
        source=dict(derivative=dict(loss=native_loss(x,target,directions),loss_derivative=derivative),
            rows=[dict(coefficient=c,loss=native_loss(y,target,directions))
                  for c,y in zip((.125,.25,.5,1.),endpoints,strict=True)])
        rows=decompose(x,u,endpoints,target,directions,source)
        errors=check_rows(x,u,endpoints,target,directions,rows,derivative)
        assert max(errors)<1e-12
        if name=='translation':
            assert all(abs(row['transport_remainder'])<1e-12 for row in rows)
            assert all(abs(row['functional_remainder']-row['coefficient']**2*.04)<1e-12 for row in rows)
        if name=='zero':assert all(row['endpoint_target_fraction'] is None for row in rows)
        records.append(dict(name=name,max_error=max(errors)))
    return dict(passed=True,controls=records)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--input');parser.add_argument('--output',required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--primary-summary')
    args=parser.parse_args();assert not Path(args.output).exists()
    torch.set_num_threads(1);base.verify_sources();math=controls()
    if args.check_only:
        dump(args.output,dict(check_only=True,**math));print('METRIC_REMAINDER_CONTROLS_PASS');return
    assert args.input
    root=Path(args.input);result=read(root/'summary.json')
    assert result['original_primary_gate_unchanged'] and result['new_model_evaluations']==0
    expected_banks,expected_points=(1,64) if result['witness'] else (8,4096)
    if not result['witness']:
        assert args.primary_summary and sha(Path(args.primary_summary))==result['original_primary_summary_sha256']
        primary=read(Path(args.primary_summary))
        assert primary['primary_full_scale_explained']==result['original_primary_result']
    assert len(result['cases'])==8 and len(result['source_bindings'])==8
    errors=[];comparisons=0;identities=set()
    for item in result['cases']:
        identity=(item['case'],item['split']);assert identity not in identities;identities.add(identity)
        folder=Path(item['source_folder']);summary=read(folder/'summary.json');binding=read(folder/'binding.json')
        assert result['source_bindings'][str(folder/'summary.json')]==sha(folder/'summary.json')
        assert len(item['banks'])==len(summary['banks'])==expected_banks
        assert summary['witness']==result['witness']
        file=Path(binding['assets']['file']);assert sha(file)==binding['assets']['sha256']
        with np.load(file) as z:inputs={key:z[key].copy() for key in z.files}
        directions=inputs['directions'].astype(float);xs=[];us=[];ends=[];targets=[]
        for index,(bank,original) in enumerate(zip(item['banks'],summary['banks'],strict=True)):
            assert bank['bank']==original['bank']==index and original['points']==expected_points
            path=folder/original['file'];assert sha(path)==original['sha256']
            with np.load(path) as z:x,u,endpoints=(z[key] for key in ('native','tangent','endpoints'))
            target=inputs[f'target_{index}'][:expected_points].astype(float)
            errors.extend(check_rows(x,u,endpoints,target,directions,bank['rows'],original['derivative']['loss_derivative']))
            comparisons+=4;xs.append(x);us.append(u);ends.append(endpoints);targets.append(target)
        errors.extend(check_rows(np.concatenate(xs),np.concatenate(us),np.concatenate(ends,axis=1),
            np.concatenate(targets),directions,item['pooled'],summary['pooled']['derivative']['loss_derivative']))
        comparisons+=4
    assert len(identities)==8 and max(errors)<1e-10
    dump(args.output,dict(passed=True,witness=result['witness'],cases=8,metric_rows=comparisons,
        independent_native_metric_max_error=max(errors),original_primary_gate_unchanged=True,
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(Path(__file__)),controls=math))
    print('METRIC_REMAINDER_NATIVE_VERIFICATION_PASS',comparisons,max(errors))


if __name__=='__main__':main()
