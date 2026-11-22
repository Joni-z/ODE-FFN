"""Locate large validation SW2 values on the ordinary Heun32 monitoring bank."""
import json
from pathlib import Path
import sys
from odd_trial import base, build_model, torch
from evaluate_screen import ev


@torch.no_grad()
def run(prep, output):
    torch.set_num_threads(1)
    ev.build_model=build_model
    rows=[]
    for dataset in ('gmm_ring8_2d','rotated_product_bimodal_16d'):
        for group, parent in [('native','coefficient-20261116/state-output/state_flow_20261116'),
                              ('odd','odd-field-20261123/train/odd_field_20261123')]:
            folder,=(prep/parent/dataset).glob('*/seed000')
            cfg=json.loads((folder/'config.json').read_text())
            cache=Path(cfg['data_root'])
            manifest=cache/'manifest.json'
            law=base.train.load_distribution(json.loads(manifest.read_text())['datasets'][dataset])
            fixed=base.train.fixed_eval_data(cache/dataset,law,law.dim,0,'cpu',4096,256,'validation')
            model,binding=ev.restore(folder,cfg,manifest,10000,'cpu')
            samples=base.train.generated_samples(model,fixed['sample_source'],32)
            score=base.train.sliced_w2(samples,fixed['sample_target'],fixed['directions'])
            archived,=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()
                       if json.loads(l)['step']==10000]
            assert abs(score-archived['sample']['sliced_w2'])<1e-6
            def radii(x):
                v=x.double().norm(dim=1)
                return [torch.quantile(v,q).item() for q in (.5,.99,1.)]
            rows.append(dict(dataset=dataset,model=group,step=10000,validation_sw2=score,
                             generated_radius_median_p99_max=radii(samples),
                             target_radius_median_p99_max=radii(fixed['sample_target']),**binding))
    base.dump(output,rows)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
