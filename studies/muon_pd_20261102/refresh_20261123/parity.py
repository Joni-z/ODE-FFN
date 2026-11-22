"""Separate sign-symmetric field error from bias that reverses incorrectly."""
from pathlib import Path
import sys
from probe import assets, base, build_model, ev, json, torch


@torch.no_grad()
def audit(prep, out):
    torch.set_num_threads(1)
    ev.build_model = build_model
    study = prep / 'coefficient-20261116'
    rows = []
    for dataset in ('gmm_ring8_2d', 'rotated_product_bimodal_16d'):
        _, fixed, bank = assets(study / 'assets', dataset, 'selection', 'cpu')
        x, t, oracle = (fixed[k] for k in ('x', 't', 'oracle'))
        folders = []
        for family in ('state_midpoint4', 'swiglu*', 'free_isru'):
            group = 'state-analysis' if family == 'state_midpoint4' else 'analysis'
            binding, = (study / group / 'selection').glob(f'{dataset}__{family}__*/binding.json')
            folders.append(Path(json.loads(binding.read_text())['source_folder']))
        for folder in folders:
            cfg = json.loads((folder / 'config.json').read_text())
            manifest = Path(cfg['data_root']) / 'manifest.json'
            law = base.train.load_distribution(json.loads(manifest.read_text())['datasets'][dataset])
            reflected = torch.as_tensor(law.velocity(-x.double().numpy(), t.double().numpy())).to(oracle)
            torch.testing.assert_close(reflected, -oracle, rtol=1e-4, atol=1e-5)
            model, identity = ev.restore(folder, cfg, manifest, 20000, 'cpu')
            pos, neg = model(x,t).double(), model(-x,t).double()
            odd, even = (pos-neg)/2, (pos+neg)/2
            raw = ((pos-oracle).square().mean()+(neg+oracle).square().mean())/2
            odd_error, even_energy = (odd-oracle).square().mean(), even.square().mean()
            torch.testing.assert_close(raw, odd_error+even_energy, atol=1e-12, rtol=1e-12)
            rows.append(dict(dataset=dataset, family=cfg['cell']['family'], paired_mse=raw.item(),
                             odd_mse=odd_error.item(), even_energy=even_energy.item(),
                             even_fraction=(even_energy/raw).item(), queries=len(x),
                             assets_sha256=bank['sha256'], **identity))
    base.dump(out,rows)


if __name__ == '__main__':
    audit(Path(sys.argv[1]), Path(sys.argv[2]))
