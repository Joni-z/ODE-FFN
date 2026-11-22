"""Score frozen-weight interventions on the established selection banks."""
import argparse
import numpy as np
from probe import assets, base, build_model, ev, intervene, json, MethodType, Path, torch

@torch.no_grad()
def run(prep, output, dataset):
    torch.set_num_threads(1)
    ev.build_model = build_model
    study = prep/'coefficient-20261116'
    config, = (study/'state-output/state_flow_20261116'/dataset).glob('*/seed000/config.json')
    cfg = json.loads(config.read_text())
    model, binding = ev.restore(config.parent,cfg,Path(cfg['data_root'])/'manifest.json',20000,'cpu')
    arrays, fixed, bank = assets(study/'assets',dataset,'selection','cpu')
    layers = list(model.ode_layers())
    assert len(layers) and all(name.split('.')[-1] in ('ode1','ode2') for name,_ in layers)
    target = torch.from_numpy(np.concatenate([arrays[f'target_{i}'] for i in range(8)])).double()
    result=[]
    for mode, depths in [('native',(4,4)),('substep',(4,4)),('entry',(4,4)),
                         ('native',(4,2)),('native',(2,4)),('native',(2,2))]:
        for name,layer in layers:
            layer.mode=mode
            layer.substeps=depths[0 if name.endswith('.ode1') else 1]
            layer.solve=MethodType(intervene,layer)
        samples=torch.cat([base.train.generated_samples(model,torch.from_numpy(arrays[f'source_{i}']),128)
                           for i in range(8)])
        sw2=base.train.sliced_w2(samples.double(),target,fixed['directions'].double())
        row=dict(dataset=dataset,split='selection',mode=mode,depths=depths,pooled_sw2=sw2,
                 assets_sha256=bank['sha256'], **binding)
        result.append(row)
        base.dump(output,result)
        print(json.dumps(row),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('prep',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('dataset')
    a=parser.parse_args()
    run(a.prep,a.output,a.dataset)
