"""Seeded algebra fixture; not a trained model or a learning result."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--save',required=True);a=p.parse_args()
rng=np.random.default_rng(0);d=8;P=rng.standard_normal((d,d));G=rng.standard_normal((d,d));C=np.diag(np.linspace(.5,3,d));M=2*np.eye(d)+np.ones((d,d));eta=.2
def root(M,inverse=False):
    values,vectors=np.linalg.eigh(M);assert (values>0).all()
    return (vectors*(1/np.sqrt(values) if inverse else np.sqrt(values)))@vectors.T
def steps(G,M):
    inv=root(M,True);H=G@inv;u,s,vh=np.linalg.svd(H,full_matrices=False)
    return -eta*G@np.linalg.inv(M)/np.linalg.norm(H),-eta*(u@vh)@inv,H,s
S=P@M@P.T;newP=P@np.linalg.inv(C);newM=C@M@C.T;Gnew=G@C.T
fro,op,H,s=steps(G,M);newfro,newop,_,_=steps(Gnew,newM)
checks=dict(increment_second_moment_invariance_max_error=float(np.max(np.abs(S-newP@newM@newP.T))),
    feature_singular_values_max_error=float(np.max(np.abs(np.linalg.svd(P@root(M),compute_uv=False)-np.linalg.svd(newP@root(newM),compute_uv=False)))),
    frobenius_functional_step_constraint_error=abs(float(np.trace(fro@M@fro.T))-eta**2),
    frobenius_step_dual_objective_error=abs(float(np.sum(G*fro))+eta*np.linalg.norm(H)),
    operator_functional_step_constraint_error=abs(float(np.linalg.norm(op@root(M),2))-eta),
    operator_step_dual_objective_error=abs(float(np.sum(G*op))+eta*s.sum()),
    frobenius_coordinate_step_max_error=float(np.max(np.abs(newfro@C-fro))),
    operator_coordinate_step_max_error=float(np.max(np.abs(newop@C-op))))
assert max(checks.values())<1e-11
out=Path(a.save);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(passed=True,scope='Seed0 width8 algebra fixture, not a learning result or an admissibility test for fixed sigmoid/direct-h model families.',checks=checks,fixture=dict(P=P.tolist(),G=G.tolist(),C=C.tolist(),M=M.tolist(),eta=eta),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(checks))
