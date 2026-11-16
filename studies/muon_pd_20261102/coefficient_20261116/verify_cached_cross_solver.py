"""Validate all existing cross-solver outputs; never generate or replace samples."""
import argparse
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
    terminal = json.loads((root/'ops/TERMINAL.json').read_text())
    native = [r for r in terminal['completed'] if r['phase']=='native']
    assert len(native)==8 and all(r['actual_terminal_exit']==0 for r in native)
    assert len({(r['id'],r['split']) for r in native})==8
    hashes,commands,proofs = {},[],[]
    for row in native:
        folder = Path(row['output'])
        complete = json.loads((folder/'COMPLETE.json').read_text())
        assert complete['passed'] and not complete['witness'] and complete['steps']==[20000]
        files = [folder/f'20000-b{i}.npy' for i in range(8)]
        files += [folder/'field-20000.npy',folder/'binding.json',folder/'summary.json',folder/'native-anchor.json']
        hashes.update({str(p):base.digest(p) for p in files})
    for row in native:
        folder = Path(row['output'])
        target = out/row['split']/row['id']
        target.mkdir(parents=True)
        invocations = [
            [sys.executable,str(HERE/'cross_solver_precision.py'),'--evaluation',str(folder),'--output',str(target/'precision')],
            [sys.executable,str(HERE/'verify_cross_solver.py'),'--input',str(folder),
             '--precision',str(target/'precision'),'--output',str(target/'verified')]]
        for index,command in enumerate(invocations):
            log = target/f'command-{index}.log'
            with log.open('x') as stream:
                result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            commands.append(dict(command=command,actual_terminal_exit=result.returncode,log=str(log)))
            base.dump(out/'command-exits.json',dict(commands=commands))
            if result.returncode:raise SystemExit(result.returncode)
        proof = json.loads((target/'verified/PASS.json').read_text())
        assert proof['passed'] and not proof['witness']
        proofs.append(dict(id=row['id'],split=row['split'],file=str(target/'verified/PASS.json'),sha256=base.digest(target/'verified/PASS.json')))
        print(row['id'],row['split'],'cached verification complete',flush=True)
    assert all(base.digest(p)==digest for p,digest in hashes.items())
    base.dump(out/'PASS.json',dict(passed=True,cases=8,native_actual_exits=[0]*8,
        actual_child_exits=[r['actual_terminal_exit'] for r in commands],proofs=proofs,
        original_generation_terminal_passed=terminal['passed'],original_generation_terminal_sha256=base.digest(root/'ops/TERMINAL.json'),
        original_samples_and_outputs_unchanged=True,original_output_hashes=hashes,
        source_sha256={p.name:base.digest(p) for p in [Path(__file__),HERE/'cross_solver_precision.py',HERE/'verify_cross_solver.py']}))


if __name__=='__main__':
    main()
