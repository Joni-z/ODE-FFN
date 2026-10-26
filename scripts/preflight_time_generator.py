"""Native pipeline smoke and initialization/replay checks before time-generator training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_time_generator as runner
from synthetic.train import restore_model


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("output","data-product","prior-product"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((ROOT/"source-sha256.json").read_text())
    for name,value in hashes.items():assert runner.base.sha256(ROOT/name)==value
    dataset="rotated_product_bimodal_16d"
    config=dict(seed=0,steps=3,workers=1,device="cpu",datasets=[dataset],arms=runner.ARMS,
                data_roots={dataset:str(args.data_product.resolve())})
    runner.base.write(output/"study.json",config);runner.base.prepare_banks(config,dataset,output)
    receipts=[];initial={}
    for arm,setting in runner.ARMS.items():
        runner.train_cell(output/"study.json",output,dataset,arm,steps=3)
        folder=output/"runs"/dataset/arm
        initial[arm]=torch.load(folder/"step_0000000.pt",map_location="cpu",weights_only=False)
        state=torch.load(folder/"step_0000003.pt",map_location="cpu",weights_only=False)
        model=runner.build(16,setting["family"],setting["mode"]);restore_model(model,state)
        native=json.loads((folder/"result.json").read_text())
        assert runner.base.field_evaluation(model,output/"banks"/dataset)["oracle_mse"]==native["oracle_mse"]
        assert len(json.loads((folder/"visits_0000003.json").read_text()))==(6 if setting["family"]=="ode_k2" else 0)
        receipts.append(dict(arm=arm,input_stream_digest=state["input_stream_digest"],replay="PASS"))
    assert len({r["input_stream_digest"] for r in receipts})==1
    for family in ("ode_k2","swiglu192"):
        a,b=[initial[f"{family}_{mode}_final"] for mode in ("static","time")]
        for key,value in a["model"].items():assert torch.equal(value,b["model"][key])
        old=args.prior_product.resolve()/"runs"/dataset/(family+"_native_final")
        old0=torch.load(old/"step_0000000.pt",map_location="cpu",weights_only=False)
        for key,value in old0["model"].items():assert torch.equal(value,a["model"][key])
        for key,value in old0["runtime_buffers"].items():assert torch.equal(value,a["runtime_buffers"][key])
        state=torch.load(old/"step_0005000.pt",map_location="cpu",weights_only=False)
        model=runner.build(16,family,"none");restore_model(model,state)
        native=json.loads((old/"result.json").read_text())
        assert runner.base.field_evaluation(model,args.prior_product.resolve()/"banks"/dataset)["oracle_mse"]==native["oracle_mse"]
    for name,value in hashes.items():assert runner.base.sha256(ROOT/name)==value
    runner.base.write(output/"PASS.json",dict(status="PASS",source_sha256=hashes,native_platform=sys.platform,
        torch=torch.__version__,smoke=receipts,initial_core_and_buffers="PASS",static_time_initialization="PASS",
        previous_parent_replay="PASS"))
    print(json.dumps(dict(status="PASS",smoke=4,previous_parent_replays=2)))
