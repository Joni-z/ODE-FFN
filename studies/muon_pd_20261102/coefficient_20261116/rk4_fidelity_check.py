"""Run bounded restoration/reference checks for both fixed solvers and tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from rk4_trial import HERE, base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    bindings = json.loads((HERE.parents[2]/
        'docs/research-assessments/2026-11-16-ode-mechanism/rk4-control-bindings.json').read_text())
    records = []
    for task, dataset in [('ring', 'gmm_ring8_2d'), ('product', 'rotated_product_bimodal_16d')]:
        row = next(r for r in bindings['references'] if
                   r['dataset'] == dataset and r['family'] == 'state_midpoint4')
        for solver in ['midpoint', 'rk4']:
            old = solver == 'midpoint'
            cell = dataset+'__state_rk4_2__adamw_0.0003'
            folder = Path(row['folder']) if old else root/'author-check/resume/state_rk4_control_20261116'/dataset/cell/'seed000'
            evaluation = Path(row['evaluations']['selection']['summary_file']).parent if old else root/('author-check/evaluation-'+dataset)
            target = out/(task+'-'+solver)
            commands = [
                [sys.executable, str(HERE/'rk4_fidelity.py'), '--folder', str(folder),
                 '--evaluation', str(evaluation), '--data-root', '/Users/mr.z/ode-workspace-prep/data/synthetic',
                 '--assets', bindings['assets_root'], '--output', str(target),
                 '--step', str(20000 if old else 16), '--witness'],
                [sys.executable, str(HERE/'verify_rk4_fidelity.py'), '--input', str(target),
                 '--output', str(out/(task+'-'+solver+'-verified'))]]
            for index, command in enumerate(commands):
                log = out/f'{task}-{solver}-{index}.log'
                with log.open('x') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                records.append(dict(command=command, actual_terminal_exit=result.returncode, log=str(log)))
                base.dump(out/'command-exits.json', dict(commands=records))
                if result.returncode:
                    raise SystemExit(result.returncode)
            native = json.loads((target/'COMPLETE.json').read_text())
            proof = json.loads((out/(task+'-'+solver+'-verified')/'PASS.json').read_text())
            assert native['all_points_qualified'] and proof['all_points_qualified']
            assert len(proof['layers']) == 6 and all(r['points'] == 8 for r in proof['layers'])
        with np.load(out/(task+'-midpoint')/'inputs.npz') as a, np.load(out/(task+'-rk4')/'inputs.npz') as b:
            for key in ['x', 'time', 'source', 'target', 'bank', 'pair']:
                assert np.array_equal(a[key], b[key]), key
    base.dump(out/'PASS.json', dict(passed=True, cases=4, layers_per_case=6,
        inputs_per_layer=8, actual_child_exits=[r['actual_terminal_exit'] for r in records],
        common_inputs_exact=True, source_sha256={p.name:base.digest(p) for p in
            [Path(__file__), HERE/'rk4_fidelity.py', HERE/'verify_rk4_fidelity.py']}))
    print(json.dumps(dict(passed=True, cases=4, actual_child_exits=[0]*8)), flush=True)


if __name__ == '__main__':
    main()
