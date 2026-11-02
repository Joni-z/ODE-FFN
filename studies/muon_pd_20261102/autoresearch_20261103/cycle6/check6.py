"""Budget-only actual-entry prefix and native checkpoint continuation witness."""
import argparse,json,subprocess,sys
from pathlib import Path
from trial6 import base,build_model
from prefix import numerical_equal,torch


def verify(device):
    base.verify_sources();torch.set_num_threads(1);counts={}
    for family in ['pd_direct_raw_k1','swiglu64','swiglu128','swiglu192']:
        torch.manual_seed(0);model=build_model(16,family).to(device);counts[family]=model.parameter_count()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    return dict(passed=True,device=device,torch=torch.__version__,parameters=counts,new_model_code=False,budget=80000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--dataset');p.add_argument('--smoke-output');a=p.parse_args();result=verify(a.device)
    if a.data_root:
        root=Path(a.smoke_output);root.mkdir(parents=True,exist_ok=False);grid=json.loads(Path(__file__).with_name('grid.json').read_text())
        assert a.dataset in ['rotated_product_bimodal_16d','gmm_ring8_2d'];ds=a.dataset;rows=[]
        for cell in [c for c in grid['cells'] if c['dataset']==ds]:
            family=cell['family'];parent_runner=base.HERE/'run.py' if cell['parent_group']=='old' else Path(__file__).resolve().parents[1]/'cycle3/trial3.py'
            def run(runner,identifier,output,steps,log,resume=None):
                command=[sys.executable,str(runner),'--cell',identifier,'--data-root',a.data_root,'--output',str(output),'--device',a.device,'--steps',str(steps)]
                if resume:command+=['--resume',str(resume)]
                with (root/log).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
            run(parent_runner,cell['parent_id'],root/'parent',8,f'{family}-parent.log')
            run(Path(__file__).with_name('trial6.py'),cell['id'],root/'long',8,f'{family}-long.log')
            parent_experiment='muon_pd_20261102' if cell['parent_group']=='old' else 'professor_autoresearch_cycle3_20261103'
            parent=root/'parent'/parent_experiment/ds/cell['parent_id']/'seed000';new=root/'long'/'professor_autoresearch_cycle6_20261103'/ds/cell['id']/'seed000'
            left=torch.load(parent/'checkpoints/step_0000008.pt',map_location='cpu',weights_only=False);right=torch.load(new/'checkpoints/step_0000008.pt',map_location='cpu',weights_only=False);numerical_equal(left,right)
            run(Path(__file__).with_name('trial6.py'),cell['id'],root/'long',16,f'{family}-resume.log',new/'checkpoints/step_0000008.pt')
            state=torch.load(new/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False);assert state['step']==state['optimizer']['count']==16
            rows.append(dict(family=family,native_old_new_eight_step_prefix_exact=True,native_resume_step=16))
        result['native']=rows;result['dataset']=ds
    base.dump(a.output,result);print(json.dumps(result))
