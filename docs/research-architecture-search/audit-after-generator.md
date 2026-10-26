# Audit after the additive-generator screen

Date: 2026-10-24. This audit reads the completed studies and their frozen source/checkpoints. It does not launch training, replace a checkpoint, or revise any completed screen. One additional frozen-feature calculation is explicitly marked below. Seed 0 remains the training protocol.

## Diagnosis and priorities

**The strongest remaining explanation is a learned-feature/optimization-efficiency gap, rather than a demonstrated shortage of scalar sign freedom, a bad final output fit, or runaway spectra.** A function-preserving coordinate change already altered fitting substantially; multiple local family changes have improved product16 while retaining a large gap to cheaper SwiGLU. The latest frozen-feature probe finds that a better linear readout does not close that gap. This narrows where to look, but is not a global expressivity lower bound.

The next useful actions are:

1. Resolve the concrete surrounding-initialization mismatch found below with the narrowly defined common-backbone control. Do not call current ODE-versus-SwiGLU comparisons exactly tensor-paired. Keep the existing within-ODE conclusions.
2. Use complete-network, residual-relevant controller responses to decide whether the separate controller MLPs are an efficient use of the block's compute. Normalize response magnitude before attributing a difference to direction or subspace. A pointwise rank relaxation is not the realized parameter tangent space.
3. If that evidence supports a design, target **shared feature computation between P and the controller**, with a genuine reduction in matrix work and a generic self-gated reference. This is different from reusing one rate tensor or sharing the two controllers, which Stage5 already rejected. It is a conditional research question, not an accepted architecture or a reason to launch another family immediately.

Continuing to change one diagonal gain, identity placement, or Taylor order without identifying its full-network limitation is unlikely to resolve the large product16 deficit. A useful architecture needs either meaningfully better learned features at comparable cost, or comparable quality at substantially lower cost.

## 1. The latest rejection is not a threshold accident

From [Stage15's exact table](analysis-additive-generator/comparison.csv):

| Product16, 5000 updates | Vector mul | Vector add | Small SwiGLU | Full SwiGLU |
|---|---:|---:|---:|---:|
| Primary oracle MSE | .266155 | .183215 | .090831 | .108234 |
| Fixed-time Jacobian error | 50.00194 | 37.85010 | 22.59628 | 25.27259 |
| Validation SW2 | .120646 | .107673 | .098223 | .094624 |
| Parameters | 83,664 | 83,664 | 71,376 | 83,598 |
| Dense linear MACs | 81,984 | 81,984 | 69,696 | 81,792 |
| Recorded training interval, seconds | 29.11 | 29.33 | 17.52 | 20.08 |

The add arm improves its paired parent by 31.2%, but still has about twice the smaller SwiGLU's field error. On Ring8 it slightly worsens its paired parent (.019930→.020283), and both exceed small SwiGLU (.015184). The wall intervals include training-time evaluation and concurrent CPU execution; they are not isolated latency benchmarks. Nevertheless, these records do not exhibit a cheaper-quality tradeoff.

The stated rule—at least 10% better field error than both baselines on a nonlinear task, at most 5% regression on other primary tasks, and derivative/SW2 safeguards—is an engineering selection rule, not a theorem implied by the research objective. Requiring a Jacobian margin on every task is stronger than requiring a practical generation-quality/compute gain, and the best metric across two baselines can form an envelope not realized by one model. These distinctions should be stated when interpreting future borderline results. **They do not rescue Stage15:** the candidate substantially loses field fit and costs more than small SwiGLU. Do not relax completed rules after inspecting the outcome.

Likewise, 5000 equal updates are not equal wall time. They give the ODE the same data exposure despite its greater cost. The current failure cannot be dismissed as the baseline having been allotted more compute. Longer fixed budgets previously improved Ring8 vector ODE but failed the nonlinear16D transfer; a new extension needs a new reason rather than a favorable training-curve segment.

## 2. Strong existing evidence and contradictions

| Explanation | Current evidence | Remaining scope |
|---|---|---|
| Conditional label noise alone prevents ODE fitting | [Stage10](population-noise-readout.md): product16 exact-mean ODE .183114 versus best SwiGLU .012699; Ring8 .002249 versus .000925. | Noise is not sufficient to close the fixed-budget gap. This does not prove an architecture expressivity bound. |
| Static K2 provides a beneficial default parameterization | [Static stage](evidence/static-summary.md): Ring8 free-B field error .028901 versus fixed-K2 .076519, with fewer MACs and equal parameters. | Free B changes the globally feasible matrix set as well as optimization. |
| Clock coordinates do not matter | [Stage11](clock-coordinates-readout.md): same initial function/class/cost, product16 .295241→.196109; Ring8 .019438→.017280. | Parameter coordinates and the optimizer response matter. Neither tested scale is a useful final architecture. |
| Remove the middle activation or one controller | [Activation](composition-readout.md) and [placement](gate-placement-readout.md) diagnostics reject a simple redundancy story; second-solve derivatives are especially important on rotated16D. | Sharing a controller is different from sharing its computed features with P. The latter remains untested. |
| Remove internal identities everywhere | [Stage14](analysis-identity-location/analysis-report.md): vector II improves product16 but still loses; scalar II switches from helpful under old initialization to harmful under function-matched initialization. | Initialization and learned response interact with identity placement; no universal removal rule. |
| Positive scalar-clock sign restriction explains the network's failure | [Stage15](analysis-additive-generator/analysis-report.md): scalar add slightly helps Ring8 but worsens product16; vector add improves product16 yet remains far behind. | The exact one-dimensional witness separates restricted local map families. Dense P, embeddings and composition can circumvent it. It is not a lower bound for VelocityNet. |
| A covariance-shaped hidden energy ensures appropriate generation | [Stage9](covariance-readout.md): its sufficient internal descent condition holds while quality fails. | Correct internal mathematics does not imply a useful inductive bias in data coordinates. |
| Large input condition numbers identify the specific bottleneck | [Projection audit](analysis-projection-geometry/analysis-report.md): both models are ill-conditioned; standardized residual-alignment shares overlap. | Neither P-only whitening nor its impossibility follows. |

These conclusions leave feature construction and the optimizer's use of it unresolved. They do not license accumulating the partially helpful coordinate, identity and additive changes into one architecture: such a combination would obscure the reason for any gain.

## 3. New frozen-feature probe: the product16 gap survives an optimal linear head

Question fixed before this calculation: does the final learned feature representation already contain an adequate velocity approximation that the current linear output head fails to extract?

For each final model, capture the width64 vector entering `model.output`, append a bias column, and solve a least-squares readout against the saved population oracle. Fit on the first 256 anchors at each of t=.1,.3,.5,.7,.9 (1280 rows total); evaluate on the remaining 256 per time. The same source-index partition applies to every time and model. NumPy `lstsq` uses fixed `rcond=1e-10`, without a ridge sweep. All four fitted feature matrices have numerical rank64, including the appended bias column; LayerNorm supplies a linear relation.

The five times have equal weight. These values differ from the primary uniform-time oracle MSE and cannot be substituted into its screen. The diagnostic bank was already exposed in prior analysis; the second partition is **not a new independent confirmation set**. Population oracle labels are privileged diagnostic supervision. No new readout was written into a model or accepted as a deployable method.

| Task | Arm | Existing head, evaluation MSE | OLS head, evaluation MSE | OLS calibration MSE |
|---|---|---:|---:|---:|
| Ring8 | Vector mul | .01580671 | .00752623 | .00655590 |
| Ring8 | Vector add | .01599076 | .00725038 | .00760537 |
| Ring8 | Small SwiGLU | .01724558 | .00394604 | .00335176 |
| Ring8 | Full SwiGLU | .01398691 | .00443612 | .00395141 |
| Product16 | Vector mul | .25750513 | .26225284 | .21894357 |
| Product16 | Vector add | .17524795 | .17468333 | .14775430 |
| Product16 | Small SwiGLU | .08692406 | .08716911 | .07474454 |
| Product16 | Full SwiGLU | .10636142 | .10737613 | .09091371 |

On product16 the simple readout intervention barely changes evaluation fit; calibration and evaluation both preserve the gap. It is therefore unpersuasive to explain the additive block's residual as only a poorly fitted last linear layer. Its learned features are less useful for this finite linear readout than SwiGLU's features. This remains conditional on the current trained representation: the architecture could learn different features with another optimization trajectory.

Ring8 exhibits a different pattern: every model's readout benefits substantially from this privileged refit, but SwiGLU remains better after the same intervention. Optimizer/output-head effects exist and cannot be mistaken for an ODE-specific remedy.

The biggest product16 errors in this partition occur at t=.7: vector mul/add existing-head MSE .739068/.501390 versus small SwiGLU .252518. After OLS they remain .730109/.491494 versus .255405. This is consistent with the earlier state/time-resolved geometry evidence. It does not isolate which block or controller created those features.

Source hashes were checked against each study's provenance. Every used NPZ hash was checked against its diagnostic summary, and the model's recomputed predictions matched the saved arrays at 1e-10 tolerances. The exact calculation source is included at the end of this file, so the extra measurement is reproducible without changing any checkpoint.

## 4. A real control gap: shared backbone architecture was not identical initialization

I separately loaded Stage15 product16 initial checkpoints. Vector mul and vector add have exactly identical values for all 22 non-FFN state tensors. Against either SwiGLU, six of those values differ: `blocks.{0,1,2}.condition.weight` and `.bias`. The largest differences in the three condition matrices are approximately .246–.248. Input/time embedding tensors, norms and the zero output head agree. The source constructs each block's FFN before its FiLM conditioning layer; different FFN parameter draws shift the global generator before the condition layer is initialized.

This does **not** invalidate the paired mul/add mechanism result. It does mean that external baseline comparisons used the same backbone *architecture* and data protocol, but not literally identical surrounding initialization. A zero initial velocity masks this difference in an initial-output check. There is no evidence yet that it explains the twofold product16 gap; neither discarding the old results nor declaring the gap fixed is warranted.

At the parent's request, a narrowly scoped helper now exists in [synthetic/initialization.py](../../synthetic/initialization.py). `copy_common_backbone` verifies the recorded checkpoint hash and all non-FFN state/runtime-buffer names and shapes before any copy, copies actual runtime constants including nonpersistent Fourier frequencies, and leaves FFN tensors and RNG states untouched. It is opt-in and has no import-time or default-model effect. [Tests](../../tests/test_common_backbone.py) check complete surrounding pairing, atomic rejection of invalid sources, and unchanged functions/gradients/three clipped Adam updates under identity application. No training or runtime source was changed by this audit.

The proposed initialization-only control should be described as a fairness/causal-attribution control. It is not a rescue of a rejected additive candidate, does not alter training seed, and should retain both practical baseline sizes. Its actual integration and registration belong to the parent task.

## 5. What must stay tied to the original objective

The current screens fairly answer a bounded question: which FFN learns these uniform-time, direct-velocity, independent-Gaussian synthetic tasks at the declared budget? They do not fully answer whether a block improves JiT.

The checked-in [JiT denoiser](../../src/denoiser.py) emits clean data, converts it to velocity, and computes velocity MSE; its time sampling is logistic-normal, with a terminal denominator clamp. Thus it would be incorrect to contrast toy velocity loss with an unweighted clean-data loss and call that the project's current JiT objective. Direct clean-data output can change finite-capacity/initialization behavior even when the algebraically converted loss agrees away from the clamp. The current image `ode` registry also refers to a different normalized single-layer block. A future image bridge needs the exact intended block, output parameterization, time law and loss declared together.

The product16 population is a deliberately discriminating analytic task with independent latent signs and varying sharpness; its covariance variation is not uniformly high effective rank at every time. Its known latent basis is evaluation information, not a model input. A useful design should explain how it learns the relevant features without hard-coding the synthetic basis. These tasks can falsify broad claims, but they do not establish that every candidate must dominate every toy metric to be useful for images.

At the same time, the current candidate loses nonlinear fitting, derivatives and cost, so a domain mismatch is not a reason to move it to expensive image experiments now. First complete the common-backbone control and the feature-response attribution. If a redesigned controller shares useful feature computation, compare it with an ordinary self-gated block at the same cost; if that generic block explains the gain, the claim should concern efficient parameterization rather than numerical ODE accuracy.

## Exact source of the frozen-feature calculation

Executed with `/Users/mr.z/ode-workspace-prep/.venv-test/bin/python` on the Mac CPU, one Torch thread. This script is recorded from the executed command after the exploratory measurement; it is not retrospectively called a preregistered experiment. Its stdout reports checkpoint hashes, both errors, feature rank, weight norms and every per-time result.

<!-- frozen-feature-source-begin -->
```python
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
root=Path('/Users/mr.z/ode-workspace-prep/architecture-search/additive_generator_20261024_m1')
sys.path.insert(0,str(root/'source'))
from synthetic.models import VelocityNet
from synthetic.train import restore_model
T=(.1,.3,.5,.7,.9);torch.set_num_threads(1)
rows=[]
for ds,studyname in [('gmm_ring8_2d','additive_generator_v1_20261024_m1'),('rotated_product_bimodal_16d','additive_generator_product16_20261024_m1')]:
 study=root/'output'/studyname
 prov=json.loads((study/'provenance.json').read_text())
 for p,digest in prov['source_sha256'].items():assert hashlib.sha256((root/'source'/p).read_bytes()).hexdigest()==digest
 for label in ('vector_mul','vector_add','swiglu_budget','swiglu'):
  cfg=json.loads((study/'configs'/f'{label}.json').read_text()); kind=next(a['model'] for a in prov['plan']['arms'] if a['label']==label)
  cp=study/'runs'/cfg['experiment_id']/ds/kind/'seed000/checkpoints/step_0005000.pt'
  state=torch.load(cp,map_location='cpu',weights_only=False)
  model=VelocityNet(16 if ds.startswith('rotated') else 2,kind,**cfg['model_options']).double().eval();restore_model(model,state)
  features=[];targets=[];preds=[];fitmask=[]
  for t in T:
   diag=study/'diagnostics'/ds/label; summary=json.loads((diag/'summary.json').read_text())
   path=diag/f'common-{t:.1f}.npz'
   assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['arrays_sha256'][path.name]
   a=np.load(path);x=torch.from_numpy(a['x']);captured=[]
   hook=model.output.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().numpy().copy()))
   with torch.no_grad():out=model(x,torch.full((len(x),1),t,dtype=torch.float64)).numpy()
   hook.remove();np.testing.assert_allclose(out,a['prediction'],atol=1e-10,rtol=1e-10)
   features.append(np.column_stack([captured[0],np.ones(len(x))]));targets.append(a['oracle']);preds.append(out);fitmask.extend(np.arange(len(x))<256)
  h=np.concatenate(features);y=np.concatenate(targets);pred=np.concatenate(preds);mask=np.array(fitmask)
  w,resid,rank,singular=np.linalg.lstsq(h[mask],y[mask],rcond=1e-10)
  fit=h@w
  row=dict(dataset=ds,label=label,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),feature_rank=int(rank),fit_n=int(mask.sum()),eval_n=int((~mask).sum()),current_cal_mse=float(np.mean((pred[mask]-y[mask])**2)),ols_cal_mse=float(np.mean((fit[mask]-y[mask])**2)),current_eval_mse=float(np.mean((pred[~mask]-y[~mask])**2)),ols_eval_mse=float(np.mean((fit[~mask]-y[~mask])**2)),head_weight_norm=float(np.linalg.norm(model.output.weight.detach().numpy())),ols_weight_norm=float(np.linalg.norm(w[:-1])),per_time_eval=[])
  for ti,t in enumerate(T):
   sl=slice(ti*512+256,(ti+1)*512); row['per_time_eval'].append(dict(time=t,current=float(np.mean((pred[sl]-y[sl])**2)),ols=float(np.mean((fit[sl]-y[sl])**2))))
  rows.append(row)
print(json.dumps(rows,indent=2))
```
<!-- frozen-feature-source-end -->

Frozen-feature script SHA256 (UTF-8 code block including its final newline): `24978450cbe3faa6a7cf0f5b7555d58b2803ce4618da7d89faed65608a80b5e8`.
