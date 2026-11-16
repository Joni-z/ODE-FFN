"""Untrained CUDA parity/derivative controls and fixed throughput probes on145."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coefficient_20261116'))
from sharing_dispatch import eligible  # noqa:E402; stdlib-only selector


def query(fields: str) -> str:
    return subprocess.check_output(['nvidia-smi','--query-'+fields,
        '--format=csv,noheader,nounits'],text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rows = list(csv.reader(io.StringIO(query('gpu=index,uuid,memory.used'))))
    apps = list(csv.reader(io.StringIO(query('compute-apps=gpu_uuid,pid'))))
    occupied = {r[0].strip() for r in apps if len(r)>1 and r[1].strip().isdigit()}
    idle = eligible(rows,occupied)
    if not idle:
        print('NO_IDLE_PERMITTED_GPU',flush=True)
        raise SystemExit(75)
    physical,uuid = idle[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        assert os.environ.get(key) == '1'
    # Import only after restricting CUDA to the selected permitted UUID.
    import numpy as np
    import torch
    from gpu_resolution import exclusive_gpu
    from hidden_defect_transport import model_response,forced_model
    from check_hidden_defect_transport import tensor_hashes,close
    from state_trial import build_model

    def guard():exclusive_gpu(uuid)
    guard()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    spec_path = HERE.parents[2]/'docs/research-assessments/2026-11-16-ode-mechanism/hidden-defect-cuda-environment.json'
    spec = json.loads(spec_path.read_text())
    assert (platform.python_version(),torch.__version__,np.__version__) == (
        spec['python'],spec['torch'],spec['numpy'])
    spec_hash = hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
    torch.cuda.reset_peak_memory_stats()

    def make_fixture(dim,points=8):
        torch.manual_seed(0)
        model = build_model(dim,'state_midpoint4').double().eval()
        with torch.no_grad():
            weight = .02*torch.sin(torch.arange(model.output.weight.numel(),dtype=torch.float64))
            model.output.weight.copy_(weight.reshape_as(model.output.weight))
        x = .7*torch.cos(torch.arange(points*dim,dtype=torch.float64).reshape(points,dim)+.1)
        t = torch.linspace(.1,.9,points,dtype=torch.float64)[:,None]
        return model,x,t

    results = []
    with torch.no_grad():
        for dim in (2,16):
            guard()
            cpu,x,t = make_fixture(dim)
            cpu_hashes = tensor_hashes(cpu)
            gpu = copy.deepcopy(cpu).cuda()
            xc,tc = x.cuda(),t.cuda()
            hashes = tensor_hashes(gpu)
            assert cpu_hashes == hashes
            methods = [layer.solve.__func__ for _,layer in gpu.ode_layers()]
            native_before = gpu(xc,tc).clone()
            primal,tangent = model_response(gpu,xc,tc,16,'midpoint','rk4')
            cpu_primal,cpu_tangent = model_response(cpu,x,t,16,'midpoint','rk4')
            portability = dict(primal=close(primal.cpu(),cpu_primal,1e-10,1e-9),
                               tangent=close(tangent.cpu(),cpu_tangent,1e-10,1e-9))
            e = torch.tensor(0.,dtype=torch.float64,device='cuda')
            jvp_p,jvp_z = torch.func.jvp(
                lambda value:forced_model(gpu,xc,tc,16,value,'midpoint','rk4'),
                (e,),(torch.ones_like(e),))
            jvp = dict(primal=close(primal,jvp_p,1e-11,1e-11),
                        tangent=close(tangent,jvp_z,1e-11,1e-11))
            finite = []
            for epsilon in (1e-4,1e-5):
                e = torch.tensor(epsilon,dtype=torch.float64,device='cuda')
                plus = forced_model(gpu,xc,tc,16,e,'midpoint','rk4')
                minus = forced_model(gpu,xc,tc,16,-e,'midpoint','rk4')
                finite.append(dict(epsilon=epsilon,
                    error=close(tangent,(plus-minus)/(2*e),2e-7,2e-4)))
            _,zero = model_response(gpu,xc,tc,16,'midpoint','midpoint')
            assert not torch.count_nonzero(zero)
            reverse_p,reverse_z = model_response(gpu,xc,tc,16,'rk4','midpoint')
            reverse = dict(primal=close(reverse_p,primal,1e-12,0),
                            tangent=close(reverse_z,-tangent,1e-12,0))
            norm = float(torch.linalg.vector_norm(tangent))
            assert norm>1e-10
            assert hashes == tensor_hashes(gpu) and cpu_hashes == tensor_hashes(cpu)
            assert methods == [layer.solve.__func__ for _,layer in gpu.ode_layers()]
            assert torch.equal(native_before,gpu(xc,tc))
            guard()
            results.append(dict(dim=dim,points=8,portability=portability,jvp=jvp,
                finite_differences=finite,zero_exact=True,sign_reversal=reverse,
                nonzero_tangent_norm=norm,tensors_methods_unchanged=True,native_replay_exact=True))
            print('CUDA_DEFECT_FIXTURE_PASS',dim,flush=True)
        benchmarks = []
        for points in (256,4096):
            guard()
            cpu,x,t = make_fixture(16,points)
            torch.cuda.synchronize();start = time.perf_counter()
            gpu = copy.deepcopy(cpu).cuda();xc,tc = x.cuda(),t.cuda()
            torch.cuda.synchronize();transfer_seconds = time.perf_counter()-start
            start = time.perf_counter()
            cpu_result = model_response(cpu,x,t,16,'midpoint','rk4')
            cpu_seconds = time.perf_counter()-start
            torch.cuda.synchronize();start = time.perf_counter()
            gpu_result = model_response(gpu,xc,tc,16,'midpoint','rk4')
            torch.cuda.synchronize();gpu_seconds = time.perf_counter()-start
            errors = [close(value.cpu(),reference,1e-10,1e-9)
                      for value,reference in zip(gpu_result,cpu_result,strict=True)]
            guard()
            benchmarks.append(dict(points=points,reference_steps=16,
                cpu_seconds=cpu_seconds,cuda_seconds=gpu_seconds,
                transfer_seconds=transfer_seconds,compute_speedup=cpu_seconds/gpu_seconds,
                speedup_including_transfer=cpu_seconds/(gpu_seconds+transfer_seconds),
                primal_error=errors[0],tangent_error=errors[1]))
            print('CUDA_DEFECT_THROUGHPUT',points,cpu_seconds,gpu_seconds,flush=True)
    result = dict(passed=True,scope='untrained portability and throughput only',
        physical_gpu=physical,gpu_uuid=uuid,gpu_name=torch.cuda.get_device_name(0),
        peak_memory_bytes=torch.cuda.max_memory_allocated(),environment_spec_hash=spec_hash,
        rows=results,benchmarks=benchmarks,learned_checkpoints_used=0,data_banks_used=0,
        training_runs=0,new_generation_samples=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observer_sha256=hashlib.sha256((HERE/'hidden_defect_transport.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('CUDA_HIDDEN_DEFECT_PREPARATION_PASS',flush=True)


if __name__ == '__main__':main()
