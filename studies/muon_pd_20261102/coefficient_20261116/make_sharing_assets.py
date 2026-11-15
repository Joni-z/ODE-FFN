"""Copied response-screen asset generator; new predeclared streams only."""
import argparse,json
from pathlib import Path
import numpy as np
from coefficient_trial import HERE,base
import sys
sys.path.insert(0,str(HERE.parent/'order_response_20261115'))
from evaluate_order import parent as ev
from synthetic.data import fm_batch

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
 for dataset,version in [('gmm_ring8_2d','v1'),('rotated_product_bimodal_16d','product16_v1')]:
  cache=base.verify_data(a.data_root,dict(dataset=dataset,data_version=version))
  law=base.train.load_distribution(json.loads((cache/'manifest.json').read_text())['datasets'][dataset])
  for split,stream in [('selection',2026112000),('confirmation',2026112100)]:
   rng=np.random.default_rng(stream+2000)
   directions=rng.standard_normal((256,law.dim)).astype(np.float32)
   directions/=np.linalg.norm(directions,axis=1,keepdims=True)
   arrays=ev.bank_arrays(law,directions,4096,8,stream)
   rng=np.random.default_rng(stream+1000)
   x,t,y=fm_batch(law.sample(rng,4096),rng)
   arrays.update(field_x=x,field_t=t,field_y=y,field_oracle=law.velocity(x,t).astype(np.float32))
   file=out/dataset/(split+'.npz');file.parent.mkdir(exist_ok=True)
   np.savez_compressed(file,**arrays)
   rows.append(dict(dataset=dataset,split=split,stream=stream,file=str(file),sha256=base.digest(file),
                    arrays={k:ev.array_hash(v) for k,v in arrays.items()},data_manifest_sha256=base.digest(cache/'manifest.json')))
 base.dump(out/'manifest.json',dict(passed=True,source_sha256=base.digest(__file__),rows=rows))
 print(json.dumps(dict(passed=True,output=str(out))))

if __name__=='__main__':main()
