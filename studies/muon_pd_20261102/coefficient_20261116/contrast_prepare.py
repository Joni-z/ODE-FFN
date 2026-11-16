"""Bind unchanged references and verify native zero-step pairing before launch."""
import argparse
import json
from pathlib import Path
from contrast_trial import HERE, TAG, base, torch


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['root', 'study', 'witness']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root, study, witness = map(Path, [a.root, a.study, a.witness])
    assert not (study/'anchors.json').exists()
    ready = json.loads((witness/'PASS.json').read_text())
    assert ready['passed'] and ready['native_continuations'] == ready['real_evaluator_witnesses'] == 2
    for name, digest in ready['source_sha256'].items():
        assert base.digest(HERE/name) == digest
    old = json.loads((root/'sharing-anchors.json').read_text())
    assets = root/'sharing-assets'
    assert base.digest(assets/'manifest.json') == old['assets_manifest_sha256']
    references = []
    for ref in old['cases']:
        folder = Path(ref['folder'])
        cfg = json.loads((folder/'config.json').read_text())
        assert (folder/'ADAPTER_COMPLETE').exists() and cfg['steps'] == 20000 and cfg['seeds'] == [0]
        names = ['config.json']+[f'checkpoints/step_{s:07d}.pt' for s in [0, 15000, 20000]]
        row = dict(ref, family=cfg['cell']['family'], dataset=cfg['cell']['dataset'],
            bound_files={name: base.digest(folder/name) for name in names}, evaluations={})
        for name, digest in ref.get('bound_files', {}).items():
            assert base.digest(folder/name) == digest
        for split in ['selection', 'confirmation']:
            evaluation = root/'sharing-analysis'/split/ref['id']
            assert json.loads((evaluation/'COMPLETE.json').read_text())['passed']
            binding = json.loads((evaluation/'binding.json').read_text())
            assert binding['assets']['sha256'] == base.digest(binding['assets']['file'])
            row['evaluations'][split] = dict(folder=str(evaluation), bound_files={f.name: base.digest(f)
                for f in evaluation.iterdir() if f.is_file() and f.suffix in ['.json', '.npy']})
        references.append(row)
    new_cases = []
    for cell in json.loads((HERE/'contrast-grid.json').read_text())['cells']:
        folder = study/'output'/TAG/cell['dataset']/cell['id']/'seed000'
        assert not folder.exists()
        new_file = witness/'resume'/TAG/cell['dataset']/cell['id']/'seed000/checkpoints/step_0000000.pt'
        new = torch.load(new_file, weights_only=False, map_location='cpu')
        parent = next(r for r in references if r['dataset'] == cell['dataset'] and r['family'] == 'state_midpoint4')
        old_file = Path(parent['folder'])/'checkpoints/step_0000000.pt'
        anchor = torch.load(old_file, weights_only=False, map_location='cpu')
        for name, value in anchor['model'].items():
            assert torch.equal(value, new['model'][name]), name
        for name in set(new['model'])-set(anchor['model']):
            assert name.endswith('stage_delta') and torch.count_nonzero(new['model'][name]) == 0
        assert torch.equal(new['data_rng'], anchor['data_rng'])
        new_cases.append(dict(id=cell['id'], dataset=cell['dataset'], folder=str(folder),
            initial_shared_model_exact=True, initial_data_rng_exact=True,
            reference_initial_checkpoint_sha256=base.digest(old_file)))
    base.dump(study/'anchors.json', dict(passed=True, references=references, new_cases=new_cases,
        assets=str(assets), assets_manifest_sha256=base.digest(assets/'manifest.json'),
        readiness_sha256=base.digest(witness/'PASS.json'), source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True, reference_cases=8, new_cases=2, initial_pairing_exact=True)))


if __name__ == '__main__':
    main()
