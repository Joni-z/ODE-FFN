"""Same-process dense-work and CPU forward/backward timing; no optimization."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from rk4_trial import base, torch, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    assert not out.exists()
    torch.set_num_threads(1)
    base.verify_sources()
    rows = []
    for dimension in [2, 16]:
        models = {}
        for family in ['state_midpoint4', 'state_rk4_2']:
            torch.manual_seed(0)
            models[family] = build_model(dimension, family).cpu()
        for key in models['state_midpoint4'].state_dict():
            assert torch.equal(models['state_midpoint4'].state_dict()[key],
                               models['state_rk4_2'].state_dict()[key])
        x = torch.linspace(-1., 1., 512*dimension).reshape(512, dimension)
        t = torch.linspace(.05, .95, 512)[:, None]
        original = torch.nn.functional.linear
        costs, times = {}, {key:dict(forward=[], forward_backward=[]) for key in models}
        for family, model in models.items():
            count = [0]
            def linear(value, weight, bias=None):
                count[0] += value.numel()//value.shape[-1]*weight.numel()
                return original(value, weight, bias)
            with torch.no_grad(), patch('torch.nn.functional.linear', side_effect=linear):
                model(x, t)
            costs[family] = count[0]//len(x)
            for _ in range(10):
                model.zero_grad(set_to_none=True)
                model(x, t).square().mean().backward()
        assert len(set(costs.values())) == 1
        for repeat in range(100):
            families = list(models)
            if repeat % 2:
                families.reverse()
            for family in families:
                model = models[family]
                with torch.no_grad():
                    before = time.perf_counter()
                    model(x, t)
                    times[family]['forward'].append(time.perf_counter()-before)
                model.zero_grad(set_to_none=True)
                before = time.perf_counter()
                model(x, t).square().mean().backward()
                times[family]['forward_backward'].append(time.perf_counter()-before)
        for family, model in models.items():
            rows.append(dict(dimension=dimension, family=family, batch=512,
                parameters=model.parameter_count(), dense_macs_per_example=costs[family],
                timings=times[family], median_seconds={key:float(np.median(value))
                    for key,value in times[family].items()}))
    result = dict(passed=True, rows=rows, repeats=100, warmup=10,
        alternating_order=True, optimizer_steps=0, threads=1,
        source_sha256=base.digest(__file__), torch=torch.__version__)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True, rows=[{k:v for k,v in r.items() if k!='timings'}
        for r in rows])), flush=True)


if __name__ == '__main__':
    main()
