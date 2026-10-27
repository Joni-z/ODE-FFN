"""Independent data decode, literal FFN and active whole-image gradient checks."""
from __future__ import annotations
import argparse,hashlib,json,pickle,sys,tarfile
from pathlib import Path
import numpy as np
import torch

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
a=ap.parse_args();root=a.root;data=a.data
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
sys.path.insert(0,str(root/'source'));from image_model import build,ffn_cost
p=json.loads((root/'preflight.json').read_text());assert p['status']=='PASS'
for name,h in json.loads((root/'bindings.json').read_text()).items():assert sha(root/name)==h
manifest=json.loads((data/'manifest.json').read_text());archive=data/'cifar-10-python.tar.gz'
assert hashlib.file_digest(archive.open('rb'),'md5').hexdigest()=='c58f30108f718f92721af3b95e74349a'
assert sha(archive)==manifest['archive_sha256']
for name,h in manifest['files'].items():assert sha(data/name)==h
images=np.load(data/'images.uint8.npy',mmap_mode='r');assert images.shape==(50000,3,32,32) and images.dtype==np.uint8
with tarfile.open(archive,'r:gz') as tar:
    for i in range(1,6):
        content=tar.extractfile(f'cifar-10-batches-py/data_batch_{i}').read()
        batch=pickle.loads(content,encoding='bytes')[b'data']
        np.testing.assert_array_equal(batch,images[(i-1)*10000:i*10000].reshape(10000,3072))
with np.load(data/'split.npz') as split:
    train,validation=split['train'],split['validation']
    assert (len(train),len(validation))==(45000,5000)
    np.testing.assert_array_equal(np.sort(np.concatenate((train,validation))),np.arange(50000))
    assert not np.intersect1d(train,validation).size
    np.testing.assert_array_equal(np.concatenate((train,validation)),np.random.default_rng(0).permutation(50000))
torch.set_num_threads(1);generator=torch.Generator().manual_seed(0);rows=[]
for family in ('ode_k1','ode_k2','swiglu86','swiglu129'):
    model=build(root/'source/jit',family);cost=ffn_cost(model)
    stored=[r for r in p['rows'] if r['family']==family];assert len(stored)==2
    for key in ('trainable_parameters','ffn_parameters_per_block','dense_FFN_forward_MACs_per_token_per_block'):
        assert all(r[key]==cost[key] for r in stored)
    if family.startswith('ode'):
        module=model.blocks[0].mlp;x=torch.randn(2,3,64,generator=generator)
        def literal(layer,value):
            duration=.2+.8*torch.sigmoid(layer.t_func(value)/10)
            total=value;term=value
            for i in range(layer.orders):term=torch.nn.functional.linear(term,layer.proj.weight)*duration/(i+1);total=total+term
            return total
        expected=literal(module.ode2,torch.nn.functional.silu(literal(module.ode1,x)))
        torch.testing.assert_close(module(x,torch.randn(2,64,generator=generator)),expected,atol=0,rtol=0)
        # Activate existing readout/AdaLN gates solely for this gradient witness.
        with torch.no_grad():
            model.final_layer.linear.weight.copy_(torch.linspace(-.02,.03,model.final_layer.linear.weight.numel()).reshape_as(model.final_layer.linear.weight))
            for block in model.blocks:
                block.adaLN_modulation.linear.bias[128:192]=.1
                block.adaLN_modulation.linear.bias[320:384]=.1
        image=torch.randn(2,3,32,32,generator=generator);t=torch.full((2,),.4)
        value=model(image,t,torch.zeros(2,dtype=torch.long))
        matrices=[layer.proj.weight for block in model.blocks for layer in (block.mlp.ode1,block.mlp.ode2)]
        gradients=torch.autograd.grad(value.square().mean(),matrices)
        assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in gradients)
        rows.append(dict(family=family,P_matrices=len(gradients),min_P_gradient_norm=min(float(g.norm()) for g in gradients),literal_forward_bitwise=True))
# A full-row-rank48x64 patch decoder is possible: no dimension-forced
# output-null subspace follows. This is not an assertion about trained rank.
w=np.random.default_rng(0).standard_normal((48,64));assert np.linalg.matrix_rank(w)==48
result=dict(status='PASS',decoded_training_images=50000,training=45000,validation=5000,
    test_batch_decoded=False,data_manifest_sha256=sha(data/'manifest.json'),source_files_checked=len(json.loads((root/'bindings.json').read_text())),
    calibration_rows=8,active_whole_image_P_checks=rows,possible_patch_decoder_rank=48,
    quality_training_launched=False,quality_results=False)
(root/'preparation-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
