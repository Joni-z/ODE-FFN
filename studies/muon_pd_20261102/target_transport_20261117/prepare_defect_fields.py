"""Copy existing actual tensors and original field arrays; generate no data."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from population_resolution import read,dump,sha


def main():
    p=argparse.ArgumentParser()
    for name in ('portable','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from common import ev
    original=read(a.portable/'portable.json');fields=read(a.assets/'manifest.json')
    shutil.copy2(a.portable/'portable.json',a.output/'portable.json')
    for case in original['cases']:
        file=a.portable/case['tensor_file'];assert sha(file)==case['tensor_sha256']
        shutil.copy2(file,a.output/file.name)
    rows=[]
    for row in fields['rows']:
        file=Path(row['file']);assert sha(file)==row['sha256']
        with np.load(file) as z:arrays={k:z[k].copy() for k in ('field_x','field_t','field_oracle')}
        hashes={k:ev.array_hash(v) for k,v in arrays.items()}
        assert all(hashes[k]==row['arrays'][k] for k in arrays)
        assert all(len(v)==4096 for v in arrays.values())
        name=row['dataset']+'-'+row['split']+'-fields.npz'
        np.savez_compressed(a.output/name,**arrays)
        rows.append(dict(dataset=row['dataset'],split=row['split'],file=name,sha256=sha(a.output/name),
            arrays=hashes,original_file=str(file),original_sha256=row['sha256']))
    protocol=Path(__file__).resolve().parents[3]/'docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-protocol.md'
    dump(a.output/'fields.json',dict(rows=rows,portable_sha256=sha(a.output/'portable.json'),
        original_assets_manifest_sha256=sha(a.assets/'manifest.json'),protocol_sha256=sha(protocol),
        source_sha256=sha(__file__),new_draws=0,new_training=0))
    print('ACTUAL_DEFECT_FIELD_ASSETS',len(rows),flush=True)


if __name__=='__main__':main()
