"""Bounded adapter: archived FM trainer + official Muon. No new training loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'reference'))
import numpy as np
import torch
from synthetic import train
from synthetic.models import VelocityNet
from vendor.muon import SingleDeviceMuon

SPECTRAL_STEPS = {1, 10, 100, 500, 1000, 2500, 5000, 10000, 15000, 20000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def build_model(dim, family, **unused):
    """Paired non-FFN tensors, including nonpersistent Fourier frequencies."""
    state = torch.get_rng_state()
    reference = VelocityNet(dim, 'ode_original', width=64, depth=3, orders=2)
    after_reference = torch.get_rng_state()
    if family == 'scalar_k2':
        return reference
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(state)
        if family in ('vector_dp_k2', 'vector_pd_k2'):
            model = VelocityNet(dim, 'ode_vector', width=64, depth=3,
                                gate_hidden=32, gate_reference_hidden=64, orders=2)
            if family == 'vector_pd_k2':
                # The copied recurrence already implements repeated P D exactly.
                # Its historical constructor guard restricted the earlier K1 study.
                for _, layer in model.ode_layers():
                    layer.rate_placement = 'before_projection'
        elif family in ('swiglu128', 'swiglu192'):
            model = VelocityNet(dim, 'swiglu', width=64, depth=3,
                                ffn_hidden=int(family.removeprefix('swiglu')))
        else:
            raise ValueError(family)
    source = dict(reference.named_parameters())
    buffers = dict(reference.named_buffers())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if '.ffn.' not in name:
                param.copy_(source[name])
        for name, buf in model.named_buffers():
            if '.ffn.' not in name:
                buf.copy_(buffers[name])
    torch.set_rng_state(after_reference)
    return model


def projection_parameters(model):
    return [(n, p) for n, p in model.named_parameters()
            if '.ffn.' in n and p.ndim == 2 and
            (n.endswith('.proj.weight') or n.endswith('.w12.weight') or n.endswith('.w3.weight'))]


def spectral(tensor):
    x = tensor.detach().double().cpu().numpy()
    s = np.linalg.svd(x, compute_uv=False)
    energy = float(np.sum(s*s))
    probability = s / s.sum() if s.sum() else np.zeros_like(s)
    positive = probability[probability > 0]
    out = dict(singular_values=s.tolist(), frobenius=float(np.sqrt(energy)),
               stable_rank=float(energy / s[0]**2) if s[0] else 0.,
               entropy_rank=float(np.exp(-(positive*np.log(positive)).sum())) if s.sum() else 0.)
    if x.shape[0] == x.shape[1]:
        eig = np.linalg.eigvals(x)
        out.update(eigenvalues=[[float(z.real), float(z.imag)] for z in eig],
                   nonnormality=float(np.linalg.norm(x.T@x-x@x.T)/(energy+1e-12)))
    return out


class ProjectionOptimizer:
    """Two native optimizers with disjoint groups and serializable state."""
    def __init__(self, model, name, lr, log_path=None):
        self.selected = projection_parameters(model)
        ids = {id(p) for _, p in self.selected}
        if len(ids) != len(self.selected) or not ids:
            raise ValueError('projection group is empty or duplicated')
        rest = [p for p in model.parameters() if id(p) not in ids]
        if len(rest)+len(ids) != len(list(model.parameters())):
            raise ValueError('optimizer parameter partition failed')
        params = [p for _, p in self.selected]
        if name == 'muon':
            self.projections = SingleDeviceMuon(params, lr=lr, momentum=.95, weight_decay=0)
        elif name == 'adamw':
            self.projections = torch.optim.AdamW(params, lr=lr, weight_decay=0)
        else:
            raise ValueError(name)
        self.aux = torch.optim.AdamW(rest, lr=3e-4, weight_decay=0)
        self.name, self.count, self.log_path = name, 0, log_path

    def zero_grad(self, **kwargs):
        self.projections.zero_grad(**kwargs)
        self.aux.zero_grad(**kwargs)

    def step(self):
        self.count += 1
        logging = self.log_path is not None and self.count in SPECTRAL_STEPS
        if logging:
            before = {n: p.detach().clone() for n, p in self.selected}
            # Official Muon modifies gradients in-place; capture first.
            gradients = {n: p.grad.detach().clone() for n, p in self.selected if p.grad is not None}
        self.projections.step()
        self.aux.step()
        if logging:
            rows = {}
            for name, p in self.selected:
                row = dict(weight=spectral(p), delta=spectral(p.detach()-before[name]))
                if name in gradients:
                    row['clipped_gradient'] = spectral(gradients[name])
                momentum = self.projections.state[p].get('momentum_buffer')
                if momentum is not None:
                    row['muon_momentum'] = spectral(momentum)
                rows[name] = row
            train.json_line(self.log_path, dict(step=self.count, optimizer=self.name, matrices=rows))

    def state_dict(self):
        return dict(name=self.name, count=self.count,
                    names=[n for n, _ in self.selected],
                    projections=self.projections.state_dict(), aux=self.aux.state_dict())

    def load_state_dict(self, state):
        if state['name'] != self.name or state['names'] != [n for n, _ in self.selected]:
            raise ValueError('optimizer identity differs')
        self.projections.load_state_dict(state['projections'])
        self.aux.load_state_dict(state['aux'])
        self.count = state['count']


def verify_sources():
    for name, expected in json.loads((HERE/'source_provenance.json').read_text())['files'].items():
        if digest(HERE/name) != expected['vendored_sha256']:
            raise ValueError(f'source identity differs: {name}')


def verify_data(data_root, cell):
    folder = Path(data_root)/cell['data_version']
    expected = json.loads((HERE/'data_identity.json').read_text())[cell['data_version']]
    if digest(folder/'manifest.json') != expected['sha256']:
        raise ValueError('data manifest differs')
    for name, entry in expected['files'].items():
        if name.startswith(cell['dataset']+'/') and digest(folder/name) != entry['sha256']:
            raise ValueError(f'data bytes differ: {name}')
    return folder


def jacobian_probe(model, fixed):
    """Full entrance-gate derivatives, evaluated at actual fixed probe states."""
    captures = {}
    handles = []
    for name, layer in model.ode_layers():
        def hook(module, inputs, key=name):
            captures[key] = inputs[0][:2].detach().clone()
        handles.append(layer.register_forward_pre_hook(hook))
    with torch.no_grad():
        model(fixed['probe_x'][:2], fixed['probe_t'][:2])
    for handle in handles:
        handle.remove()
    results = {}
    with torch.enable_grad():
        for name, layer in model.ode_layers():
            values = []
            for row in captures[name]:
                jac = torch.autograd.functional.jacobian(lambda h: layer(h[None])[0], row)
                values.append(float(torch.linalg.svdvals(jac.double().cpu())[0]))
            results[name] = values
    return results


def execute(args):
    verify_sources()
    grid = json.loads((HERE/'grid.json').read_text())
    cell = next(c for c in grid['cells'] if c['id'] == args.cell)
    folder = verify_data(args.data_root, cell)
    output = Path(args.output).resolve()
    config_path = output/'configs'/f"{cell['id']}.json"
    cfg = dict(experiment_id='muon_pd_20261102', datasets=[cell['dataset']],
               models=[cell['id']], seeds=[0], data_root=str(folder.resolve()),
               steps=grid['steps'], batch_size=grid['batch_size'], learning_rate=3e-4,
               weight_decay=0., grad_clip=1., model_options={}, eval_interval=500,
               sample_interval=5000, checkpoint_interval=5000,
               milestone_steps=[1000, 2500, 5000, 10000, 15000, 20000],
               sample_count=4096, sliced_directions=256, heun_steps=32,
               sampling_split='validation', training_targets='bank_pair', cell=cell,
               grid_sha256=digest(HERE/'grid.json'))
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError('existing configuration differs')
    else:
        dump(config_path, cfg)
    run_dir = output/cfg['experiment_id']/cell['dataset']/cell['id']/'seed000'
    raw_evaluate = train.evaluate
    models = []

    def factory(dim, ignored, **kwargs):
        model = build_model(dim, cell['family'])
        models.append(model)
        return model

    def optimizer_factory(model):
        return ProjectionOptimizer(model, cell['optimizer'], cell['projection_lr'],
                                   run_dir/'update_spectra.jsonl')

    def evaluate(model, fixed, step, *rest):
        row = raw_evaluate(model, fixed, step, *rest)
        row['sampling_split'] = 'validation'
        if step == 0 or step in cfg['milestone_steps']:
            row['projection_spectra'] = {n: spectral(p) for n, p in projection_parameters(model)}
            row['layer_jacobian_sigma_max'] = jacobian_probe(model, fixed)
        return row

    train.VelocityNet, train.evaluate = factory, evaluate
    invocation = SimpleNamespace(config=str(config_path), dataset=cell['dataset'], model=cell['id'],
                                 seed=0, device=args.device, output_root=str(output),
                                 steps=args.steps, dry_run=False, resume=args.resume,
                                 optimizer_factory=optimizer_factory)
    train.run(invocation)
    # Additional final test sampling is independent of the validation curve.
    # Do it for every completed arm; never choose a checkpoint using this number.
    if args.steps is None or args.steps == grid['steps']:
        model = models[0]
        spec = json.loads((folder/'manifest.json').read_text())['datasets'][cell['dataset']]
        fixed = train.fixed_eval_data(folder/cell['dataset'], train.load_distribution(spec),
                                      spec['dim'], 0, torch.device(args.device), 4096, 256, 'test')
        test = raw_evaluate(model, fixed, grid['steps'], None, None, True, 32)
        test['sampling_split'] = 'test'
        # Oracle MSE remains on the validation field probe: label it explicitly.
        dump(run_dir/'final_test_sampling.json', test)
        generated = train.generated_samples(model, fixed['sample_source'], 32)
        np.savez_compressed(run_dir/'final_test_examples.npz',
                            generated=generated.cpu().numpy(),
                            target=fixed['sample_target'].cpu().numpy())
    tracked = {str(p.relative_to(HERE)): digest(p) for p in HERE.rglob('*')
               if p.is_file() and p.suffix in ('.py', '.json', '.patch')}
    dump(run_dir/'adapter_provenance.json', dict(source_sha256=tracked, cell=cell,
                                               python=sys.executable, pid=os.getpid()))
    (run_dir/'ADAPTER_COMPLETE').write_text('training and required final evaluation complete\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--steps', type=int, help='preflight only; formal runs omit this')
    p.add_argument('--resume')
    execute(p.parse_args())
