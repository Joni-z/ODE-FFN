"""Four bounded real evaluator witnesses; no training or quality selection."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from rk4_trial import HERE,base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root,out = Path(args.root),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    original = HERE.parent/'response_screen_20261115/evaluate_screen.py'
    candidate = HERE/'cross_solver_evaluate.py'
    def functions(file):
        return {node.name:ast.dump(node) for node in ast.parse(file.read_text()).body
                if isinstance(node,ast.FunctionDef)}
    a,b = functions(original),functions(candidate)
    assert all(a[name]==b[name] for name in ['assets','cases'])
    native = json.loads((root/'ops/TERMINAL.json').read_text())
    controls = json.loads((HERE.parents[2]/'docs/research-assessments/2026-11-16-ode-mechanism/rk4-control-bindings.json').read_text())
    commands,proofs = [],[]
    for task,dataset in [('ring','gmm_ring8_2d'),('product','rotated_product_bimodal_16d')]:
        old = next(r for r in controls['references'] if r['dataset']==dataset and r['family']=='state_midpoint4')
        new = next(r for r in native['cases'] if r['dataset']==dataset)
        for trained,row,solver in [('midpoint',old,'rk4'),('rk4',new,'midpoint')]:
            target = out/(task+'-'+trained)
            invocations = [
                [sys.executable,str(candidate),'--folder',row['folder'],'--data-root','/Users/mr.z/ode-workspace-prep/data/synthetic',
                 '--assets',controls['assets_root'],'--output',str(target),'--device','cpu','--split','selection',
                 '--steps','20000','--solver',solver,'--witness'],
                [sys.executable,str(HERE/'cross_solver_precision.py'),'--evaluation',str(target),
                 '--output',str(out/(task+'-'+trained+'-precision'))],
                [sys.executable,str(HERE/'verify_cross_solver.py'),'--input',str(target),
                 '--output',str(out/(task+'-'+trained+'-verified')),
                 '--precision',str(out/(task+'-'+trained+'-precision'))]]
            for i,command in enumerate(invocations):
                log = out/(task+'-'+trained+f'-{i}.log')
                with log.open('x') as stream:
                    result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                commands.append(dict(command=command,actual_terminal_exit=result.returncode,log=str(log)))
                base.dump(out/'command-exits.json',dict(commands=commands))
                if result.returncode:raise SystemExit(result.returncode)
            proof = json.loads((out/(task+'-'+trained+'-verified')/'PASS.json').read_text())
            assert proof['passed'] and proof['witness']
            proofs.append(proof)
            print(task,trained,'crossed witness complete',flush=True)
    base.dump(out/'PASS.json',dict(passed=True,cases=4,actual_child_exits=[r['actual_terminal_exit'] for r in commands],
        copied_asset_and_case_functions_exact=True,proofs=proofs,
        source_sha256={p.name:base.digest(p) for p in [Path(__file__),candidate,HERE/'verify_cross_solver.py']}))


if __name__=='__main__':
    main()
