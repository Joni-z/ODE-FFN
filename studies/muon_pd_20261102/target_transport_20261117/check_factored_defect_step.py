"""CPU equality of factored eager steps and original learned-network outputs."""
import argparse
from pathlib import Path
import numpy as np
from population_resolution import read,dump,sha,load_model


def main():
    p=argparse.ArgumentParser()
    for name in ('root','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    from common import torch,ev
    import hidden_defect_transport as old
    from compiled_defect_observer import coupled_step,response_with_step
    torch.set_num_threads(1);rows=[];original_function=old.variation_solve
    for case in read(a.root/'portable.json')['cases']:
        model,_=load_model(a.root,case);snapshot=ev.tensor_snapshot(model)
        methods=[layer.solve.__func__ for _,layer in model.ode_layers()]
        rng=torch.get_rng_state().clone()
        for split in ('selection','confirmation'):
            root=a.reference/(case['id']+'-'+split)
            with np.load(root/'arrays.npz') as z:
                x=torch.from_numpy(z['x'][:8]);t=torch.from_numpy(z['time'][:8])
                expected_p=torch.from_numpy(z['primal16'][:8]);expected_z=torch.from_numpy(z['tangent16'][:8])
            primal,tangent=response_with_step(model,x,t,16,case['trained_solver'],case['inference_solver'],coupled_step)
            assert torch.equal(primal,expected_p) and torch.equal(tangent,expected_z)
            with np.load(root/'continuous_tight.npz') as z:entrance=z['entrances']
            for i,(_,layer) in enumerate(model.ode_layers()):
                y=torch.from_numpy(entrance[i,:2].copy())
                z=.01*torch.cos(torch.arange(y.numel(),dtype=y.dtype)).reshape_as(y)
                matrix=layer.operator()
                with torch.no_grad():
                    ref=old.variation_solve(y,z,matrix,16,case['trained_solver'],case['inference_solver'])
                    state=torch.cat((y,z),dim=1)
                    for _ in range(16):state=coupled_step(state,matrix,1/16,case['trained_solver'],case['inference_solver'])
                assert torch.equal(state,torch.cat(ref,dim=1))
            assert old.variation_solve is original_function
            rows.append(dict(case=case['id'],split=split,network_primal_exact=True,
                network_tangent_exact=True,nonzero_incoming_variation_layers=6))
        ev.same_tensors(model,snapshot)
        assert methods==[layer.solve.__func__ for _,layer in model.ode_layers()]
        assert torch.equal(rng,torch.get_rng_state())
    dump(a.output,dict(passed=True,compiled=False,rows=rows,network_cells=8,local_layer_checks=48,
        source_sha256=sha(__file__),step_source_sha256=sha(Path(__file__).with_name('compiled_defect_observer.py')),
        original_observer_sha256=sha(Path(__file__).with_name('hidden_defect_transport.py')),
        tensors_methods_rng_unchanged=True))
    print('FACTORED_EAGER_STEP_EXACT',len(rows),flush=True)


if __name__=='__main__':main()
