"""Run one fixed-budget autoresearch cycle against the pinned synthetic evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from efficient_trial import CALLS, HERE, base, build_model, torch

sys.path.insert(0, str(HERE.parent / "equal_compute_20261122"))
from budget import Arithmetic, DOC, read

DATASETS = ("rotated_product_bimodal_16d", "gmm_ring8_2d")
THREADS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")


class SolverArithmetic(Arithmetic):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.name == "aten::neg":
            self.counts["neg"] += args[0].numel()
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


def step_flops(dataset, family, optimizer_name="adamw", projection_lr=0.0003):
    torch.manual_seed(0)
    dim = 2 if dataset == DATASETS[1] else 16
    model = build_model(dim, family)
    optimizer = base.ProjectionOptimizer(model, optimizer_name, projection_lr)
    x, y, t = torch.randn(512, dim), torch.randn(512, dim), torch.rand(512, 1)

    def step():
        optimizer.zero_grad(set_to_none=True)
        (model((1 - t) * x + t * y, t) - (y - x)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    step()
    with SolverArithmetic() as count:
        step()
    return sum(count.counts.values())


def invoke(script, args, log):
    env = dict(os.environ, **THREADS)
    with log.open("a") as out:
        print(" ".join(map(str, args)), file=out, flush=True)
        subprocess.run([sys.executable, "-u", str(HERE/script), *map(str, args)],
                       env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def closest(target, cost):
    return (target + cost//2) // cost


def model_folder(work, cell):
    return work/"train/report4_efficient_solvers"/cell["dataset"]/cell["id"]/"seed000"


def train(work, cell, steps):
    loc = model_folder(work, cell)
    endpoint = loc/"checkpoints"/f"step_{steps:07d}.pt"
    if endpoint.exists() and (loc/"ADAPTER_COMPLETE").exists():
        return loc
    command = ["--cell",cell["id"],"--data-root",work.parent/"data/synthetic","--output",work/"train",
               "--device","cpu","--steps",steps]
    candidates = sorted(p for p in (loc/"checkpoints").glob("step_*.pt") if p.name < endpoint.name)
    if candidates:
        command += ["--resume",candidates[-1]]
    invoke("efficient_trial.py", command, work/f'{cell["id"]}-s{steps}.log')
    assert endpoint.exists() and (loc/"ADAPTER_COMPLETE").exists()
    return loc


def evaluate(work, loc, phase, family, steps, dataset, split):
    target = work/phase/split/f"{dataset}__{family}"
    if not (target/"COMPLETE.json").exists():
        invoke("evaluate.py", ["--folder",loc,"--data-root",work.parent/"data/synthetic",
               "--assets",work.parent/"coefficient-20261116/assets","--output",target,
               "--device","cpu","--split",split,"--steps",steps],
               work/f"{phase}-{split}-{dataset}-{family}.log")
    done, binding = read(target/"COMPLETE.json"), read(target/"binding.json")
    assert done["passed"] and not done["witness"] and done["steps"] == [steps]
    assert binding["source_config"]["cell"]["family"] == family and binding["source_config"]["seeds"] == [0]
    assert (binding["heun_steps"], binding["banks"], binding["samples_per_bank"]) == (128, 8, 4096)
    row, = read(target/"summary.json")
    assert base.digest(loc/"checkpoints"/f"step_{steps:07d}.pt") == row["checkpoint_sha256"]
    return dict(dataset=dataset,family=family,split=split,assets_sha256=binding["assets"]["sha256"],**row)


def write_tsv(work, rows, chosen):
    tmp = work/"results.tmp"
    with tmp.open("w") as file:
        writer = csv.writer(file,delimiter="\t")
        writer.writerow(("commit","dataset","phase","updates","flops","sw2","status","variant"))
        for r in rows:
            family = r["family"]
            status = ("baseline" if family=="state_midpoint2" else
                     "advance" if family==chosen else "discard")
            writer.writerow((r["commit"][:7],r["dataset"],r["phase"],r["step"],r["flops"],
                             f'{r["pooled_sw2"]:.8f}',status,family))
    tmp.replace(work/"results.tsv")


def cycle(work):
    torch.set_num_threads(1)
    base.verify_sources()
    branch_commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
    baseline = "state_midpoint2"
    product, ring = DATASETS
    costs = {ds: {f:step_flops(ds,f) for f in (baseline,*CALLS)} for ds in DATASETS}
    short = costs[product][baseline] * 10000
    budgets = {r["dataset"]:r["target_flops"] for r in read(DOC/"budget.json")["rows"] if r["family"]=="state_midpoint4"}
    grid = { (c["dataset"],c["family"]):c for c in read(HERE/"grid.json")["cells"] }
    base.dump(work/"budget.json",dict(costs=costs, screen_flops=short, full_flops=budgets,
                                       source_commit=branch_commit, torch_version=torch.__version__))
    past = work.parent/"report4-reduced-state-20261122"
    loc, = (past/"train/report4_reduced_state"/product).glob(f"{product}__{baseline}__*/seed000")
    status = lambda info: base.dump(work/"progress.json",dict(pid=os.getpid(),**info))
    status(dict(phase="baseline", dataset=product))
    reference = evaluate(work,loc,"screen",baseline,10000,product,"selection")
    reference.update(phase="screen",flops=short,commit=branch_commit)
    print("BASELINE",reference["pooled_sw2"],flush=True)

    def screen(family):
        cost = costs[product][family]
        steps = closest(short,cost)
        loc = train(work,grid[product,family],steps)
        r = evaluate(work,loc,"screen",family,steps,product,"selection")
        r.update(phase="screen",flops=steps*cost,commit=branch_commit)
        assert r["assets_sha256"] == reference["assets_sha256"]
        print("SCREEN",family,steps,r["pooled_sw2"],flush=True)
        return r

    status(dict(phase="screen", dataset=product, targets={f:closest(short,costs[product][f]) for f in CALLS}))
    with ThreadPoolExecutor(max_workers=3) as pool:
        arms = list(pool.map(screen,CALLS))
    selected = min(arms, key=lambda r:(r["pooled_sw2"],CALLS[r["family"]]))
    family = selected["family"]
    rows = [reference,*arms]
    write_tsv(work, rows, family)
    base.dump(work/"selection.json",dict(seed=0, selected=family, rows=rows))
    status(dict(phase="full-budget", selected=family, targets={ds:closest(budgets[ds],costs[ds][family]) for ds in DATASETS}))
    print("SELECTED",family,flush=True)

    def confirm(dataset):
        cost = costs[dataset][family]
        steps = closest(budgets[dataset],cost)
        loc = train(work,grid[dataset,family],steps)
        full = []
        for split in ("selection","confirmation"):
            r = evaluate(work,loc,"full",family,steps,dataset,split)
            r.update(phase="full",flops=steps*cost,target_flops=budgets[dataset],commit=branch_commit)
            full.append(r)
            print("FULL",dataset,split,steps,r["pooled_sw2"],flush=True)
        return full

    with ThreadPoolExecutor(max_workers=2) as pool:
        endpoints = sum(pool.map(confirm,DATASETS),[])
    write_tsv(work, rows+endpoints, family)
    base.dump(work/"results.json",dict(seed=0, selected=family, screen=rows, full=endpoints))
    status(dict(phase="complete", selected=family))


def confirm_short_budget(work):
    screen = read(work/"selection.json")
    top = screen["selected"]
    product = DATASETS[0]

    def confirm(row):
        family = row["family"]
        path = work/"screen/selection"/f"{product}__{family}"
        folder = Path(read(path/"binding.json")["source_folder"])
        result = evaluate(work,folder,"screen",family,row["step"],product,"confirmation")
        print("SHORT_CONFIRMATION",family,result["pooled_sw2"],flush=True)
        return dict(result, phase="short-confirmation", flops=row["flops"])

    references = [row for row in screen["rows"] if row["family"] in ("state_midpoint2",top)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(confirm,references))
    assert len({row["assets_sha256"] for row in result}) == 1
    base.dump(work/"short-confirmation.json",result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("--confirm-short-budget",action="store_true")
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    import fcntl
    if args.confirm_short_budget:
        confirm_short_budget(args.work)
        sys.exit(0)
    with (args.work/"cycle.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cycle(args.work)
