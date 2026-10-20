"""CPU-only operator examples with explicit spectral controls; not new FM training."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic.interventions import action


def norm(p):
    return torch.linalg.matrix_norm(p, 2).item()


def controlled_family(theta, diagonal=-.25, target_norm=1.):
    """Same eigenvalues and spectral norm; vary the shape of a nilpotent chain."""
    eye = torch.eye(3, dtype=torch.float64)
    n = torch.zeros_like(eye)
    n[0, 1], n[1, 2] = math.cos(theta), math.sin(theta)
    lo, hi = 0., 4.
    assert norm(diagonal*eye + hi*n) > target_norm
    for _ in range(80):
        mid = (lo+hi)/2
        if norm(diagonal*eye + mid*n) < target_norm:
            lo = mid
        else:
            hi = mid
    return diagonal*eye + ((lo+hi)/2)*n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    matrices = []
    for b in [0., .25, .5, 1., 2., 4.]:
        p = torch.tensor([[-.25, b], [0., -.25]], dtype=torch.float64)
        matrices.append(("fixed_eigenvalues", b, p))
        matrices.append(("rescaled_norm_one", b, p/norm(p)))
    for theta in [0., math.pi/16, math.pi/8, 3*math.pi/16, math.pi/4]:
        matrices.append(("matched_eigenvalues_and_norm", theta, controlled_family(theta)))
    records = []
    for family, parameter, p in matrices:
        eye = torch.eye(len(p), dtype=torch.float64)
        eigen = torch.linalg.eigvals(p)
        entry = dict(family=family, parameter=parameter, matrix=p.tolist(),
                     spectral_radius=eigen.abs().max().item(), spectral_norm=norm(p),
                     mu=torch.linalg.eigvalsh((p+p.T)/2).max().item(),
                     stable_rank=(p.square().sum()/norm(p)**2).item(),
                     nonnormality=(torch.linalg.matrix_norm(p.T@p-p@p.T)/p.square().sum()).item(),
                     evaluations=[])
        for s in [.2, .6, 1.]:
            exact = torch.matrix_exp(s*p)
            for k in [1, 2, 4, 8, "exact"]:
                # Basis rows give the transpose of the column-vector operator.
                op = action(eye, p, torch.full((len(p),1), s, dtype=p.dtype), k).T
                entry["evaluations"].append(dict(step=s, order=k, operator_norm=norm(op),
                    relative_operator_error=norm(op-exact)/max(norm(exact),1e-12)))
        if family == "matched_eigenvalues_and_norm":
            assert abs(entry["spectral_radius"]-.25) < 1e-12
            assert abs(entry["spectral_norm"]-1) < 1e-12
        records.append(entry)
    result = dict(status="complete", device="cpu", new_training=False, records=records,
        note="The norm-rescaled 2D family changes eigenvalues. The 3D family controls both "
             "spectral radius and spectral norm; other matrix properties still co-vary. "
             "These are mathematical examples, not evidence of improved generative performance.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), operators=len(records), evaluations=len(records)*15)))


if __name__ == "__main__":
    main()
