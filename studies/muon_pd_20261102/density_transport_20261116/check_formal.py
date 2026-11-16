"""Existing derivative checks plus two actual formal-entry witnesses."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from density_ops import HERE, base, torch, symmetry


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-uuid')
    p.add_argument('--derivative-backend', choices=['coordinate', 'batched'], default='coordinate')
    a = p.parse_args()
    root, data_root, out = map(Path, [a.root, a.data, a.output])
    cases = json.loads((root/'portable.json').read_text())['cases']
    count = len(cases)
    assert count >= 2
    assert not out.exists()
    if a.device == 'cuda':
        symmetry.exclusive_gpu(a.gpu_uuid)
    # No parent CUDA allocation before the derivative subprocess has exited.
    command = [sys.executable, '-u', str(HERE/'check_density.py'), '--root', str(root), '--data', str(data_root),
               '--output', str(out/'derivatives'), '--device', a.device]
    if a.gpu_uuid:
        command += ['--gpu-uuid', a.gpu_uuid]
    subprocess.run(command, check=True)
    derivative = json.loads((out/'derivatives/COMPLETE.json').read_text())
    assert derivative['passed'] and len(derivative['models']) == count and len(derivative['entry_witnesses']) == 2
    backend_receipt = None
    if a.derivative_backend == 'batched':
        command = [sys.executable, '-u', str(HERE/'check_backend.py'), '--root', str(root), '--data', str(data_root),
                   '--output', str(out/'backend'), '--device', a.device]
        if a.gpu_uuid:
            command += ['--gpu-uuid', a.gpu_uuid]
        subprocess.run(command, check=True)
        backend = json.loads((out/'backend/COMPLETE.json').read_text())
        assert backend['passed'] and backend['cells'] == 2*count
        backend_receipt = base.digest(out/'backend/COMPLETE.json')
    module_spec = importlib.util.spec_from_file_location('actual_formal_density', HERE/'formal_density.py')
    entry = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(entry)
    witnesses = []
    for index in [0, count-1]:
        target = out/('formal-entry-'+str(index))
        args = ['--root', str(root), '--data', str(data_root), '--index', str(index), '--output', str(target),
                '--phase', 'witness', '--device', a.device, '--derivative-backend', a.derivative_backend]
        if a.gpu_uuid:
            args += ['--gpu-uuid', a.gpu_uuid]
        entry.main(args)
        done = json.loads((target/'COMPLETE.json').read_text())
        assert done['passed'] and done['witness'] and not done['formal'] and done['parameters_unchanged']
        assert len(done['rows']) == 2 and all(r['native_endpoints_exact'] for r in done['rows'])
        witnesses.append(dict(index=index, completion_sha256=base.digest(target/'COMPLETE.json')))
    receipt = dict(passed=True, device=a.device, model_cases=count, density_data_sha256=base.digest(data_root/'density-data.json'),
        derivative_backend=a.derivative_backend, backend_completion_sha256=backend_receipt,
        device_name=torch.cuda.get_device_name(0) if a.device == 'cuda' else 'CPU',
        gpu_uuid=a.gpu_uuid, formal_witnesses=witnesses, derivative_completion_sha256=base.digest(out/'derivatives/COMPLETE.json'),
        source_sha256={p.name: base.digest(p) for p in [Path(__file__), HERE/'formal_density.py', HERE/'density_ops.py', HERE/'jacobian_backend.py']})
    base.dump(out/'COMPLETE.json', receipt)
    print(json.dumps(dict(passed=True, model_cases=count, formal_witnesses=2, device=a.device, output=str(out))))


if __name__ == '__main__':
    main()
