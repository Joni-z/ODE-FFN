"""Download and verify official CIFAR-10; never decode the test batch."""
from __future__ import annotations
import argparse,hashlib,json,pickle,tarfile,time,urllib.request
from pathlib import Path
import numpy as np

URL='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MD5='c58f30108f718f92721af3b95e74349a'

def digest(path,kind='sha256'):
    with path.open('rb') as f:return hashlib.file_digest(f,kind).hexdigest()
def main(root):
    root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    part=root/'cifar-10-python.tar.gz.part';total=0;last=started
    with urllib.request.urlopen(URL,timeout=40) as response,part.open('xb') as f:
        resolved=response.url;expected=response.headers.get('Content-Length')
        for block in iter(lambda:response.read(2**20),b''):
            f.write(block);total+=len(block)
            if time.monotonic()-last>5:
                print(json.dumps(dict(state='DOWNLOADING',bytes=total,content_length=expected)),flush=True);last=time.monotonic()
    assert digest(part,'md5')==MD5,'Official MD5 mismatch; preserve partial for inspection'
    archive=part.with_suffix('');part.rename(archive)
    arrays=[];members={}
    with tarfile.open(archive,'r:gz') as tar:
        for i in range(1,6):
            name=f'cifar-10-batches-py/data_batch_{i}';data=tar.extractfile(name).read()
            members[name]=hashlib.sha256(data).hexdigest();item=pickle.loads(data,encoding='bytes')
            a=item[b'data'];assert a.shape==(10000,3072) and a.dtype==np.uint8
            arrays.append(a.reshape(-1,3,32,32))
    images=np.concatenate(arrays);assert images.shape==(50000,3,32,32)
    np.save(root/'images.uint8.npy',images)
    indices=np.random.default_rng(0).permutation(50000)
    train,validation=indices[:45000],indices[45000:]
    assert len(np.intersect1d(train,validation))==0 and len(np.unique(indices))==50000
    np.savez(root/'split.npz',train=train,validation=validation)
    manifest=dict(source=URL,resolved_url=resolved,official_md5=MD5,archive_sha256=digest(archive),
        archive_bytes=archive.stat().st_size,decoded_members=members,shape=list(images.shape),dtype='uint8',
        layout='NCHW, RGB',train=45000,validation=5000,split_rng='numpy.default_rng(0)',
        unconditional=True,class_labels_used=False,test_batch_decoded=False,
        normalization='planned: float32(image)/127.5-1, no augmentation',
        files={p.name:digest(p) for p in [root/'images.uint8.npy',root/'split.npz']},
        preparation_seconds=time.monotonic()-started)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='DATA_READY',root=str(root),**{k:manifest[k] for k in ('train','validation','archive_bytes','preparation_seconds')})),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);main(ap.parse_args().root)
