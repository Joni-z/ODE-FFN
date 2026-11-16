"""Complete cached sensitivity-tail/path diagnosis; no model calls or filtering."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(u,delta,expected_fraction):
    energy = np.sum(u*u,axis=1)
    actual = np.sum(delta*delta,axis=1)
    residual = np.sum((u-delta)**2,axis=1)
    fraction = 1-float(residual.sum()/actual.sum()) if actual.sum() else None
    if expected_fraction is not None:
        assert abs(fraction-expected_fraction)<=1e-10*(1+abs(expected_fraction))
    else:
        assert fraction is None
    order = np.argsort(energy)[::-1]
    rows = []
    for count in (1,10,int(np.ceil(.01*len(u)))):
        chosen = order[:count]
        rows.append(dict(sources=count,tangent_energy_share=float(energy[chosen].sum()/energy.sum()) if energy.sum() else None,
            actual_energy_share=float(actual[chosen].sum()/actual.sum()) if actual.sum() else None,
            residual_energy_share=float(residual[chosen].sum()/residual.sum()) if residual.sum() else None))
    qs = [50,90,99,99.9,100]
    return dict(points=len(u),endpoint_fraction=fraction,
        tangent_rms_norm=float(np.sqrt(energy.mean())),actual_rms_norm=float(np.sqrt(actual.mean())),
        norm_quantiles=dict(percentiles=qs,tangent=np.percentile(np.sqrt(energy),qs).tolist(),
                            actual=np.percentile(np.sqrt(actual),qs).tolist()),
        tangent_ranked_energy_shares=rows,largest_tangent_index=int(order[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    terminal = read(args.input/'TERMINAL.json')
    assert terminal['passed'] and not terminal['witness'] and len(terminal['cases'])==8
    args.output.mkdir(parents=True,exist_ok=False)
    cases = [];bindings = {}
    for entry in terminal['cases']:
        assert [v['actual_terminal_exit'] for v in entry['commands']]==[0,0]
        root = Path(entry['folder']);summary = read(root/'summary.json')
        assert not summary['witness'] and len(summary['banks'])==8
        assert read(root/'COMPLETE.json')['summary_sha256']==sha(root/'summary.json')
        bindings[str(root/'summary.json')] = sha(root/'summary.json')
        all_u,all_delta,banks = [],[],[]
        for index,row in enumerate(summary['banks']):
            file = root/row['file'];assert sha(file)==row['sha256']
            with np.load(file) as saved:
                u = saved['tangent'];delta = saved['endpoints'][-1]-saved['native']
            assert u.shape==delta.shape and len(u)==4096 and np.isfinite(u).all()
            banks.append(dict(bank=index,original_numerically_qualified=row['numerical']['qualified'],
                **describe(u,delta,row['rows'][-1]['endpoint_fraction'])))
            all_u.append(u);all_delta.append(delta)
        u,delta = np.concatenate(all_u),np.concatenate(all_delta)
        pooled = describe(u,delta,summary['pooled']['rows'][-1]['endpoint_fraction'])
        bank,index = divmod(pooled['largest_tangent_index'],4096)
        pooled['largest_source'] = dict(bank=bank,index=index,in_registered_first64_probe=index<64)
        pooled['registered_probe_max_tangent_norm'] = float(max(np.linalg.norm(v[:64],axis=1).max() for v in all_u))
        with np.load(root/summary['banks'][bank]['file']) as saved:
            native_path = saved['native_path'][:,index]
            tangent_path = saved['tangent_path'][:,index]
            actual_path = saved['paths'][-1,:,index]-native_path
        assert np.array_equal(tangent_path[-1],u[pooled['largest_tangent_index']])
        assert np.array_equal(actual_path[-1],delta[pooled['largest_tangent_index']])
        example = dict(selection='single largest endpoint-tangent source in the pooled case',
            bank=bank,index=index,times=np.linspace(0,1,9).tolist(),
            tangent_norm=np.linalg.norm(tangent_path,axis=1).tolist(),
            actual_displacement_norm=np.linalg.norm(actual_path,axis=1).tolist())
        cases.append(dict(case=summary['case'],split=summary['split'],banks=banks,
            original_numerically_qualified=summary['all_numerically_qualified'],pooled=pooled,
            extreme_example=example))
        print('ENDPOINT_TAIL_CASE',summary['case'],summary['split'],flush=True)
    result = dict(scope='post-completion cached descriptive diagnosis; all64banks/8pooled cases',
        cases=cases,source_bindings=bindings,new_model_calls=0,new_samples=0,
        original_gate_changed=False,source_sha256=sha(Path(__file__)))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures = args.output/'figures';figures.mkdir()
    order = ['ring_midpoint_trained','ring_rk4_trained','product_midpoint_trained','product_rk4_trained']
    names = ['Ring8 / midpoint weights','Ring8 / RK4 weights','Product16 / midpoint weights','Product16 / RK4 weights']
    colors = ['#0072B2','#D55E00']
    for kind in ('norm-quantiles','extreme-recorded-paths'):
        fig,axes = plt.subplots(2,2,figsize=(10,7))
        for axis,case,name in zip(axes.flat,order,names,strict=True):
            for split,color in zip(['selection','confirmation'],colors,strict=True):
                data = next(v for v in cases if v['case']==case and v['split']==split)
                if kind=='norm-quantiles':
                    values = data['pooled']['norm_quantiles'];xs = np.arange(5)
                    a,b = values['tangent'],values['actual']
                else:
                    values = data['extreme_example'];xs = values['times']
                    a,b = values['tangent_norm'],values['actual_displacement_norm']
                axis.plot(xs,np.where(np.array(a)>0,a,np.nan),'--o',color=color,markersize=4)
                axis.plot(xs,np.where(np.array(b)>0,b,np.nan),'-s',color=color,markersize=4)
            axis.set_title(name);axis.set_yscale('log');axis.set_ylabel('Euclidean norm')
            if kind=='norm-quantiles':
                axis.set_xticks(range(5),['50','90','99','99.9','100'])
                axis.set_xlabel('Norm percentile (categorical spacing)')
            else:axis.set_xlabel('Native sampler recorded time')
        legend = [Line2D([0],[0],color=c,label=s) for s,c in zip(['selection','confirmation'],colors,strict=True)]
        legend += [Line2D([0],[0],color='.2',linestyle='--',marker='o',label='endpoint tangent'),
                   Line2D([0],[0],color='.2',linestyle='-',marker='s',label='actual solver displacement')]
        fig.legend(handles=legend,loc='lower center',ncol=4,fontsize=9)
        fig.tight_layout(rect=(0,.07,1,1))
        for extension in ('png','pdf'):fig.savefig(figures/(kind+'.'+extension),dpi=180)
        plt.close(fig)
    (args.output/'figure-catalog.md').write_text('# Cached sensitivity diagnosis\n\n'
        '- norm-quantiles: all8pooled cases; prescribed norm percentiles use categorical spacing. '
        'Compare tail growth of the tangent and actual finite displacement. No uncertainty bars: these are fixed empirical arrays.\n'
        '- extreme-recorded-paths: the single maximum endpoint-tangent source from each complete case/split, '
        'with all9original snapshots. These are deliberately extreme examples, not representative trajectories. '
        'Raw zeros are retained in JSON; only zero-valued log-display points are omitted.\n')
    print('CACHED_ENDPOINT_TAIL_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
