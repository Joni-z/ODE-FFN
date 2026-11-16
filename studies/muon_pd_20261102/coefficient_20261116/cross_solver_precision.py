"""FP64 crossed-field reference on the evaluation's actual bound input split."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from rk4_trial import base,torch,build_model
from cross_solver_evaluate import substitute_solver
from rk4_fidelity import tensor_hashes,read,dump,sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    evaluation,out = Path(args.evaluation),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    binding = read(evaluation/'binding.json')
    assert read(evaluation/'COMPLETE.json')['passed']
    folder = Path(binding['source_folder'])
    cfg = read(folder/'config.json')
    cache = base.verify_data('/Users/mr.z/ode-workspace-prep/data/synthetic',cfg['cell'])
    state = torch.load(folder/'checkpoints/step_0020000.pt',weights_only=False,map_location='cpu')
    assert state['config_sha256']==sha(folder/'config.json')
    assert state['dataset_manifest_sha256']==sha(cache/'manifest.json')
    asset = binding['assets']
    assert sha(Path(asset['file']))==asset['sha256']
    with np.load(asset['file']) as z:
        x,t = z['field_x'].copy(),z['field_t'].copy()
    assert len(x)==4096 and asset['split']==binding['split']
    torch.manual_seed(0)
    model = build_model(x.shape[1],cfg['cell']['family']).cpu().eval()
    base.train.restore_model(model,state)
    substitute_solver(model,binding['inference_solver'])
    model.double()
    snapshot,rng = tensor_hashes(model),torch.get_rng_state().clone()
    result = []
    with torch.no_grad():
        for start in range(0,len(x),1024):
            result.append(model(torch.from_numpy(x[start:start+1024]).double(),
                                torch.from_numpy(t[start:start+1024]).double()).numpy())
    prediction = np.concatenate(result)
    assert tensor_hashes(model)==snapshot and torch.equal(rng,torch.get_rng_state())
    np.save(out/'field-fp64.npy',prediction,allow_pickle=False)
    dump(out/'PASS.json',dict(passed=True,split=binding['split'],inference_solver=binding['inference_solver'],
        source_binding_sha256=sha(evaluation/'binding.json'),asset_sha256=asset['sha256'],
        checkpoint_sha256=sha(folder/'checkpoints/step_0020000.pt'),points=4096,
        field_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        field_t_sha256=hashlib.sha256(t.tobytes()).hexdigest(),
        prediction_sha256=sha(out/'field-fp64.npy'),parameters_and_rng_unchanged=True,
        source_sha256=sha(Path(__file__))))
    print('matched split precision reference',binding['split'],len(x),flush=True)


if __name__=='__main__':
    main()
