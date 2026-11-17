"""Independent discrete-sampler derivatives and actual learned-prefix checks."""
import argparse
from pathlib import Path
import numpy as np
from population_resolution import read, dump, sha, load_model
from common import base, ev, torch
from analytic_target import JointSensitivityField, ForcedField
from hidden_defect_transport import model_response


def close(value, reference, atol=2e-7, rtol=2e-4):
    error = (value-reference).abs()
    assert bool(torch.all(error <= atol+rtol*reference.abs()))
    return float(error.max())


def check(native, crossed, analytic, source, steps, analytic_jvp=True):
    sampler = base.train.generated_samples
    state = torch.cat((source, torch.zeros_like(source), torch.zeros_like(source)), dim=1)
    field = JointSensitivityField(native, crossed, analytic, steps)
    joined = sampler(field, state, steps)
    x, ua, um = joined.chunk(3, dim=1)
    assert field.calls == 2*steps and field.primal_qualified
    assert torch.equal(x, sampler(native, source, steps))
    errors = {}
    for name, tangent, forcing in [('analytic', ua, analytic),
                                  ('measured', um, lambda x, t: crossed(x, t)-native(x, t))]:
        jvp = None
        if analytic_jvp or name == 'measured':
            zero = torch.tensor(0., dtype=source.dtype, device=source.device)
            primal, derivative = torch.func.jvp(
                lambda coefficient: sampler(ForcedField(native, forcing, coefficient), source, steps),
                (zero,), (torch.ones_like(zero),))
            jvp = close(tangent, derivative, 1e-10, 1e-9)
            assert torch.equal(primal, x)
        finite = []
        for epsilon in (1e-4, 1e-5):
            plus = sampler(ForcedField(native, forcing, epsilon), source, steps)
            minus = sampler(ForcedField(native, forcing, -epsilon), source, steps)
            finite.append(dict(epsilon=epsilon, error=close(tangent, (plus-minus)/(2*epsilon))))
        errors[name] = dict(direct_sampler_jvp=jvp, finite_differences=finite,
            analytic_nested_jvp_unavailable=bool(name == 'analytic' and not analytic_jvp))
    return dict(steps=steps, points=len(source), primal_exact=True,
                primal_jvp_error=field.primal_max_error, errors=errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(1)
    base.verify_sources()
    sampler_rows, learned_rows = [], []

    class LinearField(torch.nn.Module):
        def __init__(self, crossed=False):
            super().__init__()
            self.matrix = torch.tensor([[-.2, .3], [-.1, .1]], dtype=torch.float64)
            if crossed:
                self.matrix = self.matrix+torch.tensor([[.02, -.01], [.01, .03]], dtype=torch.float64)
            self.crossed = crossed

        def forward(self, x, t):
            return x@self.matrix.T+.03*t+(self.crossed*.02*(1-t))

    native, crossed = LinearField(), LinearField(True)
    source = torch.tensor([[.1, -.2], [.4, .3]], dtype=torch.float64)
    analytic = lambda x, t: .017*x+.013*(1-t)
    with torch.no_grad():
        for steps in (8, 128, 2048):
            sampler_rows.append(check(native, crossed, analytic, source, steps))
            print('ANALYTIC_TARGET_LINEAR_PASS', steps, flush=True)
        for case in read(a.root/'portable.json')['cases']:
            native, crossed = load_model(a.root, case)
            snapshots = [ev.tensor_snapshot(m) for m in (native, crossed)]
            methods = [layer.solve.__func__ for _, layer in native.ode_layers()]
            rng = torch.get_rng_state().clone()
            n, c = case['trained_solver'], case['inference_solver']
            def analytic(x, t):
                return model_response(native, x, t, 16, n, c)[1]
            for split, binding in case['splits'].items():
                file = a.root/binding['assets']['file']
                assert sha(file) == binding['assets']['sha256']
                with np.load(file) as data:
                    source = torch.from_numpy(data['source_0'][:2].copy()).double()
                row = check(native, crossed, analytic, source, 8, analytic_jvp=False)
                learned_rows.append(dict(case=case['id'], split=split, **row))
                print('ANALYTIC_TARGET_LEARNED_PASS', case['id'], split, flush=True)
            for model, snapshot in zip((native, crossed), snapshots, strict=True):
                ev.same_tensors(model, snapshot)
            assert methods == [layer.solve.__func__ for _, layer in native.ode_layers()]
            assert torch.equal(rng, torch.get_rng_state())
    dump(a.output, dict(passed=True, analytic_linear_rows=sampler_rows, learned_rows=learned_rows,
                       portable_sha256=sha(a.root/'portable.json'), source_sha256=sha(__file__),
                       kernel_source_sha256=sha(Path(__file__).with_name('analytic_target.py')),
                       training_seed=0, new_training=0, new_draws=0,
                       learned_witness_reference_steps=16, formal_reference_steps=256,
                       tensors_methods_rng_unchanged=True))
    print('ANALYTIC_TARGET_CPU_PASS', len(learned_rows), flush=True)


if __name__ == '__main__':
    main()
