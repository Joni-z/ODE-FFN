"""NumPy-only replay of saved common-path and oracle-switch diagnostics."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
TAG='professor_transport15_20261104'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_hash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()


def sw2(x,y,d):
    return float(np.sqrt(np.mean((np.sort(x.astype(float)@d.astype(float).T,axis=0)-np.sort(y.astype(float)@d.astype(float).T,axis=0))**2)))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--parent-runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();runtime=Path(a.runtime);parent=Path(a.parent_runtime);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    grid=json.loads((HERE/'grid.json').read_text());binding=json.loads((runtime/'common/binding.json').read_text())
    assert binding['source_sha256']==sha(HERE/'transport.py') and binding['grid_sha256']==sha(HERE/'grid.json')
    with np.load(runtime/'common/common.npz',allow_pickle=False) as f:common={k:f[k].copy() for k in f.files}
    assert {k:array_hash(v) for k,v in common.items()}==binding['array_hashes']
    assert sha(runtime/'audit/manifest.json')==binding['manifest_sha256']
    manifest=json.loads((runtime/'audit/manifest.json').read_text())
    module_spec=importlib.util.spec_from_file_location('transport_native_data',STUDY/'reference/synthetic/data.py')
    data=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=data;module_spec.loader.exec_module(data)
    law=data.GaussianMixture(**manifest['datasets']['gmm_ring8_2d']['distribution'])
    profiles,steps,switches,endpoints=[],[],[],[];all_results={};max_score_error=0.;max_identity=0.;max_oracle_error=0.
    oracle_path=common['oracle_path'].astype(float);assert oracle_path.shape==(129,4096,2)
    for cell in grid['cells']:
        folder=runtime/'output'/TAG/cell['dataset']/cell['id']/'seed000'
        assert (folder/'ADAPTER_COMPLETE').exists()
        result=json.loads((folder/'result.json').read_text());all_results[cell['id']]=result
        assert result['passed'] and result['native_bank0_exact'] and result['tensors_unchanged'] and result['cell']==cell
        assert result['producer_sha256']==sha(HERE/'transport.py') and result['grid_sha256']==sha(HERE/'grid.json')
        assert result['common_binding_sha256']==sha(runtime/'common/binding.json')
        path=np.load(folder/'learned_path.npy',allow_pickle=False).astype(float);assert path.shape==(129,4096,2) and np.isfinite(path).all()
        original=(parent/'lenovo/output/professor_validation14_20261104'/cell['dataset']/cell['source_case']/'seed000')
        original_rows=[json.loads(line) for line in (original/'rows.jsonl').read_text().splitlines()]
        row=next(r for r in original_rows if r['checkpoint_step']==cell['checkpoint_step'] and r['bank']==0 and r['heun_steps']==128)
        assert result['checkpoint']['checkpoint_sha256']==row['checkpoint_sha256']
        assert np.array_equal(path[-1],np.load(original/row['generated_file'],allow_pickle=False))
        with np.load(original/'banks.npz',allow_pickle=False) as f:
            for name,oldname in [('source','source_0'),('target','target_0'),('target2','target2_0'),('directions','directions')]:assert np.array_equal(common[name],f[oldname])
        with np.load(folder/'profiles.npz',allow_pickle=False) as f:saved={k:f[k].copy() for k in f.files}
        assert len(result['profiles'])==33 and len(result['steps'])==128 and len(result['switches'])==8
        assert np.isfinite(saved['jacobian']).all() and np.isfinite(saved['oracle_jacobian']).all()
        for index,record in enumerate(result['profiles']):
            time=index/32;assert record['time']==time
            state=oracle_path[index*4];teacher=law.velocity(state,np.full((len(state),1),time))
            assert np.allclose(teacher,saved['oracle'][index],rtol=1e-12,atol=1e-12)
            error=saved['prediction'][index].astype(float)-teacher
            radius=np.linalg.norm(state,axis=1);valid=radius>1e-8
            radial=np.divide(state,radius[:,None],out=np.zeros_like(state),where=valid[:,None]);angular=np.stack([-radial[:,1],radial[:,0]],axis=1)
            er=(error*radial).sum(axis=1);ea=(error*angular).sum(axis=1)
            metrics=dict(coordinate_mse=float(np.mean(error**2)),radial_energy=float(np.mean(er[valid]**2)),angular_energy=float(np.mean(ea[valid]**2)),radial_bias=float(np.mean(er[valid])),angular_bias=float(np.mean(ea[valid])))
            for name,value in metrics.items():assert np.isclose(value,record[name],rtol=1e-10,atol=1e-12),(cell['id'],time,name)
            assert record['valid_radius']==int(valid.sum())
            jac=saved['jacobian'][index].astype(float);uj=saved['oracle_jacobian'][index]
            for name,value in [('jacobian_norm_mean',np.linalg.svd(jac,compute_uv=False)[:,0].mean()),('symmetric_growth_mean',np.linalg.eigvalsh((jac+jac.transpose(0,2,1))/2)[:,-1].mean()),('oracle_jacobian_mse',np.mean((jac-uj)**2))]:
                assert np.isclose(value,record[name],rtol=1e-5,atol=1e-7)
            if index in [4,16,28]:
                fixture=state[:8];columns=[]
                for j in range(2):
                    delta=np.zeros_like(fixture);delta[:,j]=1e-6
                    columns.append((law.velocity(fixture+delta,np.full((8,1),time))-law.velocity(fixture-delta,np.full((8,1),time)))/2e-6)
                finite=np.stack(columns,axis=2);oracle_error=float(np.max(np.abs(finite-uj[:8])));max_oracle_error=max(max_oracle_error,oracle_error)
                assert np.allclose(finite,uj[:8],rtol=1e-5,atol=1e-6)
            profiles.append(dict(cell=cell['id'],family=cell['family'],checkpoint_step=cell['checkpoint_step'],**record))
        displacement=path-oracle_path;defect=saved['learned_step_at_oracle'].astype(float)-oracle_path[1:];response=path[1:]-saved['learned_step_at_oracle'].astype(float)
        identity=float(np.max(np.abs(displacement[1:]-response-defect)));assert identity<1e-12 and identity==result['identity_max_error'];max_identity=max(max_identity,identity)
        for index,record in enumerate(result['steps']):
            assert index==record['step']
            for key,values in [('defect_mse',defect[index]),('state_error_mse',displacement[index]),('response_mse',response[index])]:assert np.isclose(np.mean(values**2),record[key],rtol=1e-12,atol=1e-15)
            norm=np.linalg.norm(displacement[index],axis=1);mask=norm>1e-8;gain=np.linalg.norm(response[index,mask],axis=1)/norm[mask]
            assert record['gain_count']==int(mask.sum())
            for key,value in [('gain_mean',gain.mean() if len(gain) else None),('gain_median',np.median(gain) if len(gain) else None),('gain_p90',np.quantile(gain,.9) if len(gain) else None),('gain_max',gain.max() if len(gain) else None)]:
                assert record[key] is None if value is None else np.isclose(record[key],value,rtol=1e-12,atol=1e-12)
            steps.append(dict(cell=cell['id'],family=cell['family'],checkpoint_step=cell['checkpoint_step'],**record))
        for record in result['switches']:
            file=folder/record['file'];assert sha(file)==record['sha256'];values=np.load(file,allow_pickle=False);assert values.shape==(4096,2) and np.isfinite(values).all()
            score=sw2(values,common['target'],common['directions']);error=abs(score-record['sw2']);max_score_error=max(max_score_error,error);assert error<1e-10
            endpoint=float(np.mean((values.astype(float)-oracle_path[-1])**2));assert np.isclose(endpoint,record['endpoint_oracle_mse'],rtol=1e-12,atol=1e-15)
            if record['mode']=='learned':assert np.array_equal(values,path[-1])
            if record['mode']=='oracle':assert np.array_equal(values,oracle_path[-1])
            switches.append(dict(cell=cell['id'],family=cell['family'],checkpoint_step=cell['checkpoint_step'],**record))
        native=next(r for r in result['switches'] if r['mode']=='learned')
        times=np.array([r['time'] for r in result['profiles']]);mse=np.array([r['coordinate_mse'] for r in result['profiles']])
        endpoints.append(dict(cell=cell['id'],family=cell['family'],checkpoint_step=cell['checkpoint_step'],sw2=native['sw2'],endpoint_oracle_mse=native['endpoint_oracle_mse'],common_path_integrated_mse=float(np.trapezoid(mse,times)),time_early_mse=float(mse[:9].mean()),time_late_mse=float(mse[24:].mean()),identity_max_error=identity))
        print(f"Replayed {cell['id']}: {len(endpoints)}/6",flush=True)
    assert len(profiles)==198 and len(steps)==768 and len(switches)==48 and len(endpoints)==6
    write(out/'profiles.csv',profiles);write(out/'step_decomposition.csv',steps);write(out/'switches.csv',switches);write(out/'endpoints.csv',endpoints)
    (out/'reference.json').write_text(json.dumps(binding,indent=2)+'\n')
    receipt=dict(passed=True,cases=6,profile_rows=198,step_rows=768,endpoint_rows=48,native4096_anchors=6,identity_max_error=max_identity,numpy_sw2_max_error=max_score_error,actual_oracle_fd_max_error=max_oracle_error,common_arrays_exact=True,new_training_runs=0,producer_sha256=sha(HERE/'transport.py'),grid_sha256=sha(HERE/'grid.json'),analyzer_sha256=sha(__file__))
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'pd_isru_k1':'ISRU-D','free_isru':'Free ISRU','swiglu64':'Selected Swi64'}
    families=list(labels)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for column,family in enumerate(families):
        for step,color,style in [(15000,'#0072B2','-'),(20000,'#D55E00','--')]:
            r=[r for r in profiles if r['family']==family and r['checkpoint_step']==step]
            axes[0,column].plot([x['time'] for x in r],[x['coordinate_mse'] for x in r],color=color,linestyle=style,label=f'{step//1000}k')
        before=[r for r in profiles if r['family']==family and r['checkpoint_step']==15000];after=[r for r in profiles if r['family']==family and r['checkpoint_step']==20000]
        for metric,color in [('radial_energy','#0072B2'),('angular_energy','#D55E00')]:axes[1,column].plot([x['time'] for x in after],[a[metric]-b[metric] for a,b in zip(after,before)],color=color,label=metric.replace('_energy',''))
        axes[0,column].set(title=labels[family],yscale='log',ylabel='Common-path velocity MSE');axes[0,column].legend()
        axes[1,column].axhline(0,color='black',linewidth=.7);axes[1,column].set(xlabel='Global FM time',ylabel='20k minus15k error energy');axes[1,column].legend()
    fig.savefig(out/'time-and-direction.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    oracle_score=next(r['sw2'] for r in binding['oracle_rows'] if r['steps']==128)
    for ax,family in zip(axes,families):
        for step,style in [(15000,'-'),(20000,'--')]:
            for mode,color in [('learned_prefix','#0072B2'),('oracle_prefix','#D55E00')]:
                rows=sorted([r for r in switches if r['family']==family and r['checkpoint_step']==step and r['mode']==mode],key=lambda r:r['cut_step'])
                ax.plot([r['cut_step']/128 for r in rows],[r['sw2'] for r in rows],marker='o',color=color,linestyle=style,label=f'{step//1000}k '+mode.replace('_',' '))
        ax.axhline(oracle_score,color='black',linestyle=':',label='Full oracle')
        ax.set(title=labels[family],xlabel='Switch time',ylabel='SW2, same bank0 target')
    axes[1].legend(fontsize=7);fig.savefig(out/'oracle-switches.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    for ax,family in zip(axes,families):
        for step,color in [(15000,'#0072B2'),(20000,'#D55E00')]:
            r=[r for r in steps if r['family']==family and r['checkpoint_step']==step and r['gain_count']>0]
            ax.plot([x['step']/128 for x in r],[x['gain_median'] for x in r],color=color,label=f'{step//1000}k')
        ax.axhline(1,color='black',linewidth=.7);ax.set(title=labels[family],xlabel='Global FM time',ylabel='Median actual error-direction gain');ax.legend()
    fig.savefig(out/'directional-propagation.png',dpi=180);plt.close(fig)
    print(json.dumps(receipt))


if __name__=='__main__':main()
