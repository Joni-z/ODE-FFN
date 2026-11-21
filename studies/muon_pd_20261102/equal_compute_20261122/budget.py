"""Profile the archived CPU training update and set the nearest whole-update budget."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coefficient_20261116"))
from state_trial import base, build_model, torch
from torch.utils._python_dispatch import TorchDispatchMode

REPO = HERE.parents[2]
DOC = REPO / "docs/report4/equal-compute"
PREVIOUS = REPO / "docs/report4/tied-residual"
CASES = ("free_isru", "swiglu64", "swiglu192", "state_midpoint4")


def read(path):
    return json.loads(Path(path).read_text())


class Arithmetic(TorchDispatchMode):
    """Algorithmic scalar FLOPs. Views, copies, RNG and comparisons are excluded."""
    def __init__(self):
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func._schema.name.removeprefix("aten::")
        if name in ("mm", "addmm"):
            left, right = args[:2] if name == "mm" else args[1:3]
            count = 2 * left.numel() * right.shape[1]
            if name == "addmm":
                count += out.numel()
        elif name in ("add", "sub", "rsub", "mul", "div", "sqrt", "sin", "cos",
                      "reciprocal", "rsqrt", "add_", "mul_", "pow"):
            count = out.numel()
            if name == "pow" and args[1] in (0, 1):
                count = 0
            if name in ("add", "add_") and (kwargs or {}).get("alpha", 1) != 1:
                count += out.numel()
        elif name in ("lerp_", "addcmul_", "addcdiv_"):
            count = 3 * out.numel()
        elif name in ("sum", "mean"):
            count = args[0].numel() - out.numel() + (out.numel() if name == "mean" else 0)
        elif name in ("silu", "silu_backward"):
            count = (4 if name == "silu" else 8) * out.numel()
        elif name == "native_layer_norm":
            x = args[0]
            # Mean, variance, affine normalization; one reciprocal sqrt per row.
            count = 7 * x.numel() + 2 * out[1].numel()
        elif name == "native_layer_norm_backward":
            count = 10 * args[0].numel()  # input and both affine gradients
        elif name in ("linalg_vector_norm", "_foreach_norm"):
            inputs = args[0] if name.startswith("_foreach") else [args[0]]
            count = sum(2 * value.numel() for value in inputs)
        elif name == "_foreach_mul_":
            count = sum(value.numel() for value in args[0])
        elif name in ("unsqueeze", "cat", "t", "split", "detach", "expand", "view",
                      "ones_like", "stack", "clamp", "_to_copy", "transpose", "_local_scalar_dense",
                      "profiler::_record_function_enter_new", "profiler::_record_function_exit"):
            count = 0
        else:
            raise ValueError(f"Unpriced training operation {func}")
        self.counts[name] += count
        return out


def measure(spec, folder, checkpoint):
    cfg = read(folder / "config.json")
    cell = cfg["cell"]
    dim = spec["dim"]
    model = build_model(dim, cell["family"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == state["optimizer"]["count"] == 20000
    assert state["identity"] == dict(model=cell["id"], dataset=cell["dataset"], seed=0)
    assert state["config_sha256"] == base.digest(folder / "config.json")
    base.train.restore_model(model, state)
    opt = base.ProjectionOptimizer(model, cell["optimizer"], cell["projection_lr"])
    opt.load_state_dict(state["optimizer"])
    x, y, t = torch.randn(512, dim), torch.randn(512, dim), torch.rand(512, 1)
    opt.zero_grad(set_to_none=True)
    with Arithmetic() as forward:
        inp, target = (1 - t) * x + t * y, y - x
        loss = (model(inp, t) - target).square().mean()
    with Arithmetic() as backward:
        loss.backward()
    with Arithmetic() as update:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
    phases = dict(forward=forward.counts, backward=backward.counts, optimizer=update.counts)
    return {key: sum(val.values()) for key, val in phases.items()}, phases


def profile(reference, output):
    base.verify_sources()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rows = []
    for old in read(PREVIOUS / "trained-results.json")["models"]:
        family, dataset = old["family"], old["dataset"]
        if family not in CASES:
            continue
        parent = "state-analysis" if family == "state_midpoint4" else "analysis"
        binding, = (reference / parent / "confirmation").glob(f"{dataset}__{family}__*/binding.json")
        folder = Path(read(binding)["source_folder"])
        cfg = read(folder / "config.json")
        assert cfg["batch_size"] == 512 and cfg["seeds"] == [0] and cfg["steps"] == 20000
        cp = folder / "checkpoints/step_0020000.pt"
        assert old["checkpoint_sha256"] == base.digest(cp)
        assert old["config_sha256"] == base.digest(folder / "config.json")
        manifest = base.verify_data(Path(cfg["data_root"]).parent, cfg["cell"]) / "manifest.json"
        assert old["manifest_sha256"] == base.digest(manifest)
        phases, details = measure(read(manifest)["datasets"][dataset], folder, cp)
        rows.append(dict(dataset=dataset, family=family, id=cfg["cell"]["id"],
                         optimizer=cfg["cell"]["optimizer"], source_folder=str(folder),
                         config_sha256=old["config_sha256"], checkpoint_sha256=old["checkpoint_sha256"],
                         manifest_sha256=old["manifest_sha256"], phase_flops=phases,
                         step_flops=sum(phases.values()), operations=details))
    assert len(rows) == 6
    for row in rows:
        ref, = [r for r in rows if r["dataset"] == row["dataset"] and r["family"] == "state_midpoint4"]
        target = 20000 * ref["step_flops"]
        n = (target + row["step_flops"] // 2) // row["step_flops"]
        row.update(target_flops=target, total_steps=n, actual_flops=n * row["step_flops"],
                   budget_error_fraction=n * row["step_flops"] / target - 1)
        print(row["dataset"], row["family"], n, row["phase_flops"])
    base.dump(output, dict(seed=0, batch_size=512, reference_steps=20000, rows=rows,
                           torch_version=torch.__version__, profiler_sha256=base.digest(__file__)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    profile(a.reference, a.output)
