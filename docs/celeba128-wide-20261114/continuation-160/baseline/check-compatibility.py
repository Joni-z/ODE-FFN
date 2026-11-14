from pathlib import Path
import ast,hashlib,json,subprocess,sys,torch,copy,importlib.util,gc
root=Path('/scratch/zz5070/ODE_FFN');old=root/'runtimes/celeba128-20261112';new=root/'runtimes/celeba128-wide-20261114';ops=root/'operations/celeba128-wide-20261114/continuation-160/baseline'
sys.path.insert(0,str(new));from denoiser import Denoiser;from util import misc
expected='317153d1954e2723214c310dd25e93897a5030358488a01d06ff0967ba03fab1';checkpoint=root/'runs/celeba/20261112-b16-swiglu-s0/checkpoint-last.pth'
torch.set_num_threads(2);torch._dynamo.config.disable=True
r={'kind':'native_baseline_resume_compatibility','source_commit':subprocess.check_output(['git','-C',str(new),'rev-parse','HEAD'],text=True).strip(),'original_source_commit':subprocess.check_output(['git','-C',str(old),'rev-parse','HEAD'],text=True).strip(),'image_size':128,'batch':256,'checkpoint':str(checkpoint),'math_checks':{}}
def function(path,name):
 tree=ast.parse(path.read_text());matches=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name];assert len(matches)==1
 return ast.dump(matches[0],include_attributes=False)
for file,names in [('denoiser.py',['drop_labels','sample_t','forward','generate','_forward_sample','_euler_step','_heun_step','update_ema']),('engine_jit.py',['train_one_epoch','evaluate']),('util/misc.py',['add_weight_decay'])]:
 for name in names:
  assert function(old/file,name)==function(new/file,name),(file,name)
  r['math_checks'][file+':'+name]='identical AST'
for file in ['model_jit.py','util/model_util.py','util/crop.py','util/lr_sched.py']:
 assert (old/file).read_bytes()==(new/file).read_bytes(),file;r['math_checks'][file]='byte identical'
r['native_math_identical']=True
h=hashlib.sha256()
with checkpoint.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==expected;r['checkpoint_sha256']=h.hexdigest()
state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);a=copy.deepcopy(state['args']);assert state['epoch']==99 and a.ffn=='swiglu' and a.seed==0 and a.lr==5e-5 and a.lr_schedule=='constant' and a.batch_size==256
spec=importlib.util.spec_from_file_location('legacy_native_denoiser',old/'denoiser.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
torch.manual_seed(0);first=legacy.Denoiser(a);rng=torch.get_rng_state().clone();torch.manual_seed(0);second=Denoiser(a)
assert torch.equal(rng,torch.get_rng_state());assert list(first.state_dict())==list(second.state_dict())
assert all(torch.equal(t,second.state_dict()[n]) for n,t in first.state_dict().items());r['native_state_and_rng_identical']=True
assert sum(p.numel() for p in second.parameters())==130405632
first.load_state_dict(state['model'],strict=True);second.load_state_dict(state['model'],strict=True);first.eval();second.eval()
with torch.no_grad():
 x=torch.randn(2,3,128,128);t=torch.tensor([.25,.75]);labels=torch.ones(2,dtype=torch.long)
 y1=first.net(x,t,labels);y2=second.net(x,t,labels);assert torch.equal(y1,y2)
r['full_checkpoint_prediction_bitwise_identical']=True
optimizer=torch.optim.AdamW(misc.add_weight_decay(second,0),lr=5e-5,betas=(.9,.95));optimizer.load_state_dict(state['optimizer']);steps=sorted(set(int(v['step']) for v in optimizer.state.values() if 'step'in v));assert steps==[63500]
r['restored_optimizer_steps']=steps;r['finite']={k:all(torch.isfinite(t).all().item() for t in state[k].values() if torch.is_tensor(t)) for k in ['model','model_ema1','model_ema2']};assert all(r['finite'].values());assert 'rng_state'in state
ready=json.loads((root/'operations/celeba128-20261112/cuda-readiness.json').read_text());assert ready['status']=='PASS' and ready['source_commit']==r['original_source_commit'];assert ready['full_JiT_B16']['swiglu']['batch']==256
r['legacy_GPU_readiness_status']='PASS';r['legacy_GPU_readiness_source']=ready['source_commit'];r['legacy_GPU_readiness_native_profile']=ready['full_JiT_B16']['swiglu'];r['GPU_test_repeated']=False
periodic=json.loads((root/'ODE-FFN/docs/celeba128-wide-20261114/periodic-eval-check.json').read_text());assert periodic['status']=='PASS' and periodic['model_optimizer_two_EMAs_and_training_RNG_bitwise_unchanged'];r['periodic_evaluation_RNG_check']='PASS';r['periodic_evaluation_actual_FID_computed']=False
r['status']='PASS';(ops/'compatibility.json').write_text(json.dumps(r,indent=2)+'\n');print('BASELINE_CONTINUATION_COMPATIBILITY_PASS',flush=True)
