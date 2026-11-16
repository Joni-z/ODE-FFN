"""Separate endpoint-transport and target-functional errors without new sampling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squared_distance(x: np.ndarray, target: np.ndarray, directions: np.ndarray) -> float:
    residual=np.sort(x@directions.T,axis=0)-np.sort(target@directions.T,axis=0)
    return float(np.mean(residual**2))


def decompose(x: np.ndarray, tangent: np.ndarray, endpoints: np.ndarray,
              target: np.ndarray, directions: np.ndarray, source_summary: dict) -> list[dict]:
    initial=squared_distance(x,target,directions)
    assert abs(initial-source_summary['derivative']['loss'])<1e-11
    derivative=source_summary['derivative']['loss_derivative']
    rows=[]
    for actual,source in zip(endpoints,source_summary['rows'],strict=True):
        coefficient=source['coefficient']
        linear_endpoint=x+coefficient*tangent
        predicted_loss=squared_distance(linear_endpoint,target,directions)
        actual_loss=squared_distance(actual,target,directions)
        assert abs(actual_loss-source['loss'])<1e-11
        change=actual_loss-initial
        linear_change=coefficient*derivative
        transport=actual_loss-predicted_loss
        functional=predicted_loss-initial-linear_change
        closure=change-linear_change-transport-functional
        assert abs(closure)<1e-12
        rows.append(dict(coefficient=coefficient,baseline_loss=initial,
            actual_loss=actual_loss,linear_endpoint_loss=predicted_loss,
            actual_loss_change=change,linear_loss_change=linear_change,
            endpoint_metric_change=predicted_loss-initial,
            transport_remainder=transport,functional_remainder=functional,
            total_remainder=change-linear_change,closure=closure,
            endpoint_target_fraction=1-transport**2/change**2 if change else None,
            endpoint_target_sign_correct=bool(np.sign(predicted_loss-initial)==np.sign(change))))
    return rows


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--primary-summary');parser.add_argument('--witness',action='store_true')
    args=parser.parse_args();root,out=Path(args.input),Path(args.output)
    terminal=read(root/'TERMINAL.json')
    assert terminal['passed'] and terminal['witness']==args.witness
    assert len(terminal['cases'])==8 and not out.exists()
    primary=read(Path(args.primary_summary)) if args.primary_summary else None
    assert args.witness or primary is not None
    if primary is not None:
        assert len(primary['source_bindings'])==8 and len(primary['rows'])==32
    rows=[];bindings={};cases={}
    expected_count,expected_points=(1,64) if args.witness else (8,4096)
    for entry in terminal['cases']:
        assert [r['actual_terminal_exit'] for r in entry['commands']]==[0,0]
        folder=Path(entry['folder']);summary=read(folder/'summary.json');verified=read(folder/'independent.json')
        binding=read(folder/'binding.json')
        assert summary['witness']==args.witness and verified['passed']
        assert verified['summary_sha256']==sha(folder/'summary.json')
        assert len(summary['banks'])==expected_count and binding['training_seed']==0
        assert binding['lambdas']==[.125,.25,.5,1.]
        if primary is not None:
            assert primary['source_bindings'][str(folder/'summary.json')]==sha(folder/'summary.json')
        bindings[str(folder/'summary.json')]=sha(folder/'summary.json')
        asset=Path(binding['assets']['file']);assert sha(asset)==binding['assets']['sha256']
        with np.load(asset) as z:inputs={key:z[key].copy() for key in z.files}
        directions=inputs['directions'].astype(float)
        pooled=dict(x=[],u=[],endpoints=[],target=[])
        banks=[]
        for bank,row in enumerate(summary['banks']):
            assert row['bank']==bank and row['points']==expected_points
            file=folder/row['file'];assert sha(file)==row['sha256']
            with np.load(file) as z:x,u,endpoints=(z[key] for key in ('native','tangent','endpoints'))
            target=inputs[f'target_{bank}'][:expected_points].astype(float)
            values=decompose(x,u,endpoints,target,directions,row)
            banks.append(dict(bank=bank,rows=values))
            for name,value in [('x',x),('u',u),('endpoints',endpoints),('target',target)]:pooled[name].append(value)
        x,u,target=(np.concatenate(pooled[key]) for key in ('x','u','target'))
        endpoints=np.concatenate(pooled['endpoints'],axis=1)
        values=decompose(x,u,endpoints,target,directions,summary['pooled'])
        item=dict(case=summary['case'],split=summary['split'],qualified=summary['all_numerically_qualified'],
            banks=banks,pooled=values,source_folder=str(folder))
        key=(summary['case'],summary['split']);assert key not in cases;cases[key]=item
        rows.append(item)
    assert len(cases)==8
    out.mkdir(parents=True,exist_ok=False);figures=out/'figures';figures.mkdir()
    result=dict(witness=args.witness,source_bindings=bindings,cases=rows,
        original_primary_summary_sha256=sha(Path(args.primary_summary)) if primary is not None else None,
        original_primary_result=primary['primary_full_scale_explained'] if primary is not None else None,
        original_primary_gate_unchanged=True,training_runs=0,new_model_evaluations=0,
        registration='Secondary deterministic error accounting added during full execution',source_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    order=['ring_midpoint_trained','ring_rk4_trained','product_midpoint_trained','product_rk4_trained']
    names=['Ring8 / midpoint weights','Ring8 / RK4 weights','Product16 / midpoint weights','Product16 / RK4 weights']
    colors=['#0072B2','#D55E00']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ('endpoint-metric','remainder-components'):
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for axis,case,name in zip(axes.flat,order,names,strict=True):
            for split,color in zip(['selection','confirmation'],colors,strict=True):
                values=cases[(case,split)]['pooled'];xs=[0.]+[r['coefficient'] for r in values]
                series=([('actual_loss_change','-o','actual'),('linear_loss_change','--','loss tangent'),
                         ('endpoint_metric_change',':s','endpoint tangent + metric')] if kind=='endpoint-metric' else
                        [('transport_remainder','-o','transport'),('functional_remainder','--s','functional')])
                for key,style,label in series:
                    axis.plot(xs,[0.]+[r[key] for r in values],style,color=color,label=split+' '+label)
            axis.set_title(name);axis.axhline(0,color='.65',linewidth=.7)
            axis.set_xlabel('Solver interpolation lambda');axis.set_ylabel('Squared SW2 change' if kind=='endpoint-metric' else 'Squared SW2 remainder')
            axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        if args.witness:fig.suptitle('64-source execution witness; no quality conclusion',fontsize=12)
        legend=[Line2D([0],[0],color=color,label=split) for split,color in zip(['selection','confirmation'],colors,strict=True)]
        styles=([('-','o','actual'),('--',None,'loss tangent'),(':','s','endpoint tangent + metric')]
                if kind=='endpoint-metric' else [('-','o','transport'),('--','s','functional')])
        legend.extend(Line2D([0],[0],color='.2',linestyle=style,marker=marker,label=label)
                      for style,marker,label in styles)
        fig.legend(handles=legend,loc='lower center',ncol=3 if kind=='endpoint-metric' else 4,fontsize=9)
        fig.tight_layout(rect=(0,.09,1,.95 if args.witness else 1))
        for suffix in ('png','pdf'):fig.savefig(figures/(kind+'.'+suffix),dpi=180)
        plt.close(fig)
    (out/'analysis-report.md').write_text('# Endpoint transport versus target-functional remainder\n\n'+
        ('Execution witness only; no full-cohort conclusion.\n\n' if args.witness else
         'Complete secondary accounting of all four frozen cases, both splits and every bank. The original primary result remains '+str(result['original_primary_result'])+'.\n\n')+
        '![Target changes](figures/endpoint-metric.png)\n\n'
        'The dotted prediction applies the unchanged empirical target metric to X0+lambda*u. It uses no fitted coefficient. '
        'Comparison with the dashed loss tangent separates functional nonlinearity from trajectory nonlinearity.\n\n'
        '![Error components](figures/remainder-components.png)\n\n'
        'Solid curves are target error due to endpoint linearization. Dashed curves are the exact functional remainder along the linearized endpoints, including sorting changes. '
        'Their signed sum equals the original loss-tangent error. No bank, direction or layer is selected.\n')
    (out/'stats-appendix.md').write_text('# Scope\n\n'
        'This is exact descriptive error accounting, not a new significance or architecture-superiority test. '
        'Pooled and individual-bank results remain separate. Original confidence intervals and the primary gate are unchanged. '
        'No resampling, training, new model evaluation, fitted scaling or selected subset. Numerical qualifications are inherited and retained for every case.\n')
    (out/'figure-catalog.md').write_text('# Figure purposes\n\n'
        '- endpoint-metric: compare actual target changes with a linear loss and the exact metric on linearized endpoints. Determines which approximation fails.\n'
        '- remainder-components: show signed transport and functional terms. Their sum is the original residual; cancellation must not be hidden. '
        'No error bars: curves are pooled empirical values, not estimates across training runs.\n')
    print(json.dumps(dict(passed=True,witness=args.witness,cases=len(cases),output=str(out))))


if __name__=='__main__':main()
