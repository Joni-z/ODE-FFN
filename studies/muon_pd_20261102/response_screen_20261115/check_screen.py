"""CPU response algebra, full input gradients, native save/resume and evaluation."""
import argparse,importlib.util,json,subprocess,sys
from pathlib import Path
from screen_trial import HERE,TAG,RESPONSES,base,torch,build_model,response,diagonal,previous_builder
from evaluate_screen import ev

spec=importlib.util.spec_from_file_location('response_screen_check_helpers',HERE.parent/'followup_20261115/check.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def explicit(layer,h):
 a=layer.operator()*layer.step_size(h[None])[0][None,:]
 term=h;out=h
 for k in range(1,layer.orders+1):term=a@term/k;out=out+term
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);base.verify_sources();rows=[]
 for name in RESPONSES:
  for order in (1,8):
   dim=16 if order==1 else 2;family=f'screen_{name}_k{order}'
   torch.manual_seed(0);ref=previous_builder(dim,'pd_isru_k1');rng=torch.get_rng_state()
   torch.manual_seed(0);model=build_model(dim,family)
   helpers.same(ref.state_dict(),model.state_dict());helpers.same(dict(ref.named_buffers()),dict(model.named_buffers()));helpers.same(rng,torch.get_rng_state())
   assert model.parameter_count()==(56706 if dim==2 else 58512)
   model=model.double();layer=next(model.ode_layers())[1]
   for scale in (0.,.1,2.):
    h=torch.linspace(-.99,1.01,64,dtype=torch.double)*scale
    assert torch.allclose(layer(h[None])[0],explicit(layer,h),rtol=1e-11,atol=1e-11)
    if name!='softsign' or scale!=0:
     assert torch.autograd.gradcheck(lambda u:layer(u[None])[0],(h.requires_grad_(),),fast_mode=True,atol=1e-6,rtol=1e-4)
   x=torch.linspace(-3,3,121,dtype=torch.double)
   assert torch.allclose(diagonal(name,x)*x,response(name,x),atol=1e-14,rtol=1e-14)
   assert diagonal(name,torch.zeros(1,dtype=torch.double)).item()==1
   with torch.no_grad():model.output.weight.copy_(torch.linspace(-.03,.04,model.output.weight.numel()).reshape_as(model.output.weight))
   x=torch.randn(8,dim,dtype=torch.double);t=x.new_full((8,1),.3)
   model(x,t).square().mean().backward()
   assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.norm()>0 for _,p in base.projection_parameters(model))
   rows.append(dict(response=name,order=order,dim=dim,polynomial_and_gradient=True,initialization_rng_exact=True,parameters=model.parameter_count(),softsign_nonsmooth_zero_excluded=name=='softsign'))
 base.dump(out/'kernels.json',dict(passed=True,rows=rows,torch=torch.__version__,source_sha256=base.digest(__file__)))
 grid=json.loads((HERE/'grid.json').read_text());done=[]
 for name in RESPONSES:
  for order in (1,8):
   dataset='rotated_product_bimodal_16d' if order==1 else 'gmm_ring8_2d'
   c=next(c for c in grid['cells'] if c['dataset']==dataset and c['response']==name and c['order']==order)
   def folder(root):return root/TAG/dataset/c['id']/'seed000'
   def run(root,steps,label,resume=None):
    cmd=[sys.executable,str(HERE/'screen_trial.py'),'--cell',c['id'],'--data-root',a.data_root,'--output',str(root),'--device','cpu','--steps',str(steps)]
    if resume:cmd+=['--resume',str(resume)]
    with (out/(c['id']+'-'+label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   run(out/'resume',8,'8');run(out/'resume',16,'resume16',folder(out/'resume')/'checkpoints/step_0000008.pt')
   run(out/'continuous',16,'continuous16')
   left=torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False)
   right=torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt',map_location='cpu',weights_only=False)
   helpers.numerical(left,right)
   cfg=json.loads((folder(out/'resume')/'config.json').read_text());cache=base.verify_data(a.data_root,c)
   restored,identity=ev.restore(folder(out/'resume'),cfg,cache/'manifest.json',16,'cpu')
   torch.manual_seed(0);original=build_model(2 if dataset=='gmm_ring8_2d' else 16,c['family']).eval();base.train.restore_model(original,left)
   x=torch.randn(16,original.dim);t=x.new_full((len(x),1),.4)
   assert torch.equal(restored(x,t),original(x,t))
   assert torch.equal(base.train.generated_samples(restored,x,32),base.train.generated_samples(original,x,32))
   if name=='erf' and order==8:
    cmd=[sys.executable,str(HERE/'evaluate_screen.py'),'--folder',str(folder(out/'resume')),'--data-root',a.data_root,'--assets',a.assets,'--output',str(out/'evaluation-witness'),'--device','cpu','--steps','16','--witness']
    with (out/'evaluation-witness.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   done.append(dict(cell=c['id'],resume_exact=True,evaluator_restore_exact=True,**identity));base.dump(out/'progress.json',dict(completed=done))
 base.dump(out/'PASS.json',dict(passed=True,cells=done,kernel_rows=len(rows),source_sha256=base.digest(__file__)))
 print(json.dumps(dict(passed=True,cells=len(done),output=str(out))))

if __name__=='__main__':main()
