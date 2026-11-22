"""Four-cell Product16 test of training and sampling with a bounded readout."""
import json
import sys
from pathlib import Path
from odd_trial import base, build_model, torch
from evaluate_screen import ev


@torch.no_grad()
def run(prep, output):
    torch.set_num_threads(1)
    ev.build_model=build_model
    ds='rotated_product_bimodal_16d'
    rows=[]
    for bounded_training,parent in (
        (False,'odd-field-20261123/train/odd_field_20261123'),
        (True,'bounded-odd-20261123/train/bounded_odd_20261123'),
    ):
        folder,=(prep/parent/ds).glob('*/seed000')
        cfg=json.loads((folder/'config.json').read_text())
        cache=Path(cfg['data_root'])
        manifest=cache/'manifest.json'
        law=base.train.load_distribution(json.loads(manifest.read_text())['datasets'][ds])
        data=base.train.fixed_eval_data(cache/ds,law,law.dim,0,'cpu',4096,256,'validation')
        model, binding=ev.restore(folder,cfg,manifest,10000,'cpu')
        assert model.bounded_carrier is bounded_training
        snapshot=ev.tensor_snapshot(model)
        for bounded_eval in (False,True):
            model.bounded_carrier=bounded_eval
            samples=base.train.generated_samples(model,data['sample_source'],32)
            sw2=base.train.sliced_w2(samples,data['sample_target'],data['directions'])
            radius=samples.double().norm(dim=1)
            if bounded_training==bounded_eval:
                report,=[v for v in map(json.loads,(folder/'metrics.jsonl').read_text().splitlines()) if v['step']==10000]
                assert abs(sw2-report['sample']['sliced_w2'])<1e-6
            rows.append(dict(dataset=ds,step=10000,bounded_training=bounded_training,
                             bounded_sampling=bounded_eval,validation_sw2=sw2,
                             median_radius=radius.median().item(),max_radius=radius.max().item(),
                             **binding))
        ev.same_tensors(model,snapshot)
    base.dump(output,rows)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
