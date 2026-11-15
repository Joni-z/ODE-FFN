"""Test solver order at learned hidden states against the actual ODE solution."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from gpu_resolution import base,torch,load,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--families',nargs='+',default=['state_midpoint4','free_stage_midpoint4'],choices=['state_midpoint4','free_stage_midpoint4','stage_operator_midpoint4']);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);assert not out.exists();torch.set_num_threads(1);rows=[]
    for case in json.loads((root/'portable.json').read_text())['cases']:
        if case['family'] not in a.families:continue
        model=load(root,case,'cpu').double();data=arrays(root,case);captures={};hooks=[]
        for name,layer in model.ode_layers():
            def hook(m,args,key=name):captures[key]=args[0][:8].detach().clone()
            hooks.append(layer.register_forward_pre_hook(hook))
        with torch.no_grad():model(torch.from_numpy(data['field_x'][:8]).double(),torch.from_numpy(data['field_t'][:8]).double())
        for h in hooks:h.remove()
        layers=[]
        for name,layer in model.ode_layers():
            h=captures[name];shape=h.shape;matrix=layer.operator().detach().numpy()
            weights=[matrix]+[matrix+v.detach().numpy() for v in layer.stage_delta] if hasattr(layer,'stage_delta') else [matrix]
            references=[]
            for tolerance in [1e-11,1e-12]:
                value=h.numpy().reshape(-1)
                for segment,weight in enumerate(weights):
                    def drift(s,y):
                        x=y.reshape(shape);return ((x/np.sqrt(1+1.2*x*x))@weight.T).reshape(-1)
                    sol=solve_ivp(drift,(segment/len(weights),(segment+1)/len(weights)),value,method='DOP853',rtol=tolerance,atol=tolerance/10)
                    assert sol.success;value=sol.y[:,-1]
                references.append(value.reshape(shape))
            denominator=max(float(np.sqrt(np.mean(references[-1]**2))),1e-12)
            reference_shift=float(np.sqrt(np.mean((references[0]-references[1])**2))/denominator);assert reference_shift<1e-8
            errors=[]
            with torch.no_grad():
                for n in [4,8,16,32,64]:
                    value=layer.solve(h,n).numpy();error=float(np.sqrt(np.mean((value-references[-1])**2))/denominator)
                    errors.append(dict(substeps=n,relative_error=error))
            rates=[float(np.log2(max(errors[i]['relative_error'],1e-30)/max(errors[i+1]['relative_error'],1e-30))) for i in range(4)]
            layers.append(dict(layer=name,errors=errors,observed_rates=rates,reference_refinement=reference_shift,reference_segments=len(weights),
                 gamma=layer.stage_gamma.detach().tolist() if hasattr(layer,'stage_gamma') else [1.,1.,1.,1.]))
        rows.append(dict(id=case['id'],family=case['family'],layers=layers,checkpoint_sha256=case['checkpoint_sha256'],assets_sha256=case['assets_sha256']))
        base.dump(out,dict(passed=True,scope='learned-state solver verification, not generation ranking',cases=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
