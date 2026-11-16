# Optional CPU bank batching: documented native validation

Use unchanged response-screen-cpu@04f11f75 in `.aris/compute/macmini.md`.
No installation or rebuild. Keep the native sampler, model, source order,
eight banks, numerical settings and all32768samples. Only CPUbatch size
changes from32768to4096. Existing full-batch defaults and the frozen running
observer remain unchanged.

From `/Users/mr.z/ODE-FFN-upstream`, execute these FOUR native-entry commands
verbatim and sequentially. Every output directory must be absent. These
bounded512-source entries check dispatch and numerical paths; the separate
author preflight checks complete32768-source endpoints against all eight
native reference banks. Do not launch a full queue during this validation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case gmm_ring8_2d__palindrome_operator_midpoint4__adamw_0.0003 --mode outer-subset --bank-batches --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/bankwise-doc-witness/gmm_ring8_2d__palindrome_operator_midpoint4__adamw_0.0003
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case gmm_ring8_2d__alternating_operator_midpoint4__adamw_0.0003 --mode outer-subset --bank-batches --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/bankwise-doc-witness/gmm_ring8_2d__alternating_operator_midpoint4__adamw_0.0003
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case rotated_product_bimodal_16d__palindrome_operator_midpoint4__adamw_0.0003 --mode outer-subset --bank-batches --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/bankwise-doc-witness/rotated_product_bimodal_16d__palindrome_operator_midpoint4__adamw_0.0003
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case rotated_product_bimodal_16d__alternating_operator_midpoint4__adamw_0.0003 --mode outer-subset --bank-batches --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/bankwise-doc-witness/rotated_product_bimodal_16d__alternating_operator_midpoint4__adamw_0.0003
```

Require exits0/0/0/0, four COMPLETE records, bank_batches=true,512sources and
outer64/128/256/512at inner4. Report every documentation/runtime mismatch.

Then execute the independent comparison below verbatim. It checks sixteen
exact array matches to the previous actual entries and independent SW2values.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 - <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'studies/muon_pd_20261102/coefficient_20261116')
from verify_numerical import sw2
study=Path('/Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order')
root=study/'portable/sampling-models'
cases=json.loads((root/'portable.json').read_text())['cases']
rows=[];maximum=0.
for case in cases:
 folder=study/'bankwise-doc-witness'/case['id']
 done=json.loads((folder/'COMPLETE.json').read_text())
 assert done['passed'] and done['bank_batches'] and done['case']==case
 assert [(r['inner_substeps'],r['heun_steps'],r['samples']) for r in done['rows']]==[(4,s,512) for s in [64,128,256,512]]
 with np.load(root/case['assets']) as z:data={k:z[k] for k in z.files}
 for row in done['rows']:
  file=folder/row['generated_file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==row['generated_sha256']
  value=np.load(file);reference=np.load(study/'numerical-witness'/case['id']/row['generated_file'])
  assert np.array_equal(value,reference)
  error=abs(sw2(value,data['target_0'][:512],data['directions'])-row['pooled_sw2'])
  maximum=max(maximum,error);assert error<1e-10
  rows.append(dict(id=case['id'],outer=row['heun_steps'],exact_reference=True,error=error))
result=dict(passed=True,cases=4,rows=16,max_numpy_error=maximum,comparisons=rows)
with (study/'bankwise-doc-independent.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}))
PY
```

Before formal routing require full native-bank endpoint proofs for all four
checkpoints, timing, publication and source freeze. The existing queue accepts
`--bank-batches`; retain all eight full tasks and twenty unique cells. Keep
the old queue/workers until a replacement actually advances and every retained
old completed prefix is audited. Never stop for a waiting allocation.
