import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from synthetic.data import GaussianMixture, RotatedGaussianMixture, RotatedProductMixture
from synthetic.population import PopulationStream, fingerprint_batch
from synthetic.prepare import prepare
from synthetic.train import run


class PopulationStreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def distributions(self):
        rotation=np.linalg.qr(np.random.default_rng(0).normal(size=(3,3)))[0]
        mixture=GaussianMixture([[.8,-.3,.4],[-1.,.7,1.2]],[[.03,.5,1.],[.2,.1,.7]],[.3,.7])
        rotated=RotatedGaussianMixture(**mixture.to_dict(),rotation=rotation)
        product=RotatedProductMixture([.5,1.,1.5],[.04,.2,.7],rotation)
        return mixture,rotated,product

    def test_torch_oracles_match_numpy_including_endpoint_limits(self):
        for distribution in self.distributions():
            stream=PopulationStream(distribution)
            x=torch.randn((19,3),generator=torch.Generator().manual_seed(0),dtype=torch.float64)
            t=torch.linspace(0,1,19,dtype=torch.float64)[:,None]
            value=stream.velocity(x,t)
            np.testing.assert_allclose(value.numpy(),distribution.velocity(x.numpy(),t.numpy()),atol=1e-12,rtol=1e-12)
            self.assertFalse(value.requires_grad)

    def test_sampling_moments_match_distribution_and_float32_training_contract(self):
        for distribution in self.distributions():
            stream=PopulationStream(distribution)
            sample=stream.sample(100000,torch.Generator().manual_seed(0))
            self.assertEqual(sample.dtype,torch.float32)
            self.assertTrue(torch.isfinite(sample).all())
            mean,covariance=distribution.moments()
            np.testing.assert_allclose(sample.numpy().mean(axis=0),mean,atol=.025,rtol=0)
            np.testing.assert_allclose(np.cov(sample.numpy(),rowvar=False),covariance,atol=.04,rtol=0)

    def test_mean_labels_do_not_change_inputs_rng_or_fingerprint(self):
        stream=PopulationStream(self.distributions()[-1])
        final=[]
        for supervision in ('pair','mean'):
            generator=torch.Generator().manual_seed(100000)
            digest=hashlib.sha256(b'').hexdigest()
            for _ in range(17):
                x1=stream.sample(23,generator)
                x0=torch.randn(x1.shape,generator=generator)
                t=torch.rand((len(x1),1),generator=generator)
                x=(1-t)*x0+t*x1
                label=stream.velocity(x,t) if supervision=='mean' else x1-x0
                self.assertTrue(torch.isfinite(label).all())
                digest=fingerprint_batch(digest,x1,x0,t)
            final.append((digest,generator.get_state()))
        self.assertEqual(final[0][0],final[1][0])
        torch.testing.assert_close(final[0][1],final[1][1],atol=0,rtol=0)

    def test_population_resume_keeps_optimizer_stream_and_actual_inputs_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            data_cfg=json.loads(Path('configs/synthetic/data_v1.json').read_text())
            data_cfg.update(datasets=['gaussian_aniso_2d'],train_size=64,validation_size=64,test_size=64,probe_size=64)
            data_path=base/'data.json';data_path.write_text(json.dumps(data_cfg))
            prepare(data_path,base/'data')
            cfg=json.loads(Path('configs/synthetic/gates_20261021/ode_original.json').read_text())
            cfg.update(data_root=str(base/'data'),datasets=['gaussian_aniso_2d'],models=['ode_original'],
                seeds=[0],steps=3,batch_size=8,eval_interval=1,sample_interval=2,checkpoint_interval=2,
                sample_count=16,sliced_directions=8,heun_steps=2,training_targets='population_mean')
            cfg['model_options'].update(width=8,depth=1,gate_hidden=8,ffn_hidden=16)
            config=base/'config.json';config.write_text(json.dumps(cfg))
            def args(out,steps=None,resume=None):
                return SimpleNamespace(config=str(config),dataset='gaussian_aniso_2d',model='ode_original',seed=0,
                    device='cpu',output_root=str(base/out),steps=steps,resume=resume,dry_run=False)
            run(args('full'))
            run(args('resume',steps=2))
            relative=Path(cfg['experiment_id'])/'gaussian_aniso_2d/ode_original/seed000'
            folder=base/'resume'/relative
            run(args('resume',steps=3,resume=str(folder/'checkpoints/step_0000002.pt')))
            full=torch.load(base/'full'/relative/'checkpoints/step_0000003.pt',weights_only=False,map_location='cpu')
            resumed=torch.load(folder/'checkpoints/step_0000003.pt',weights_only=False,map_location='cpu')
            self.assertEqual(full['training_stream'],resumed['training_stream'])
            self.assertEqual(full['training_stream']['batches'],3)
            for name,value in full['model'].items():
                torch.testing.assert_close(value,resumed['model'][name],atol=0,rtol=0)
            metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            self.assertIsNone(metrics[-1]['train_pair_mse'])
            self.assertIsNotNone(metrics[-1]['train_target_mse'])
            cfg.update(training_targets='population_pair',experiment_id=cfg['experiment_id']+'_pair')
            config.write_text(json.dumps(cfg))
            run(args('pair'))
            pair_folder=base/'pair'/cfg['experiment_id']/'gaussian_aniso_2d/ode_original/seed000'
            pair=torch.load(pair_folder/'checkpoints/step_0000003.pt',weights_only=False,map_location='cpu')
            self.assertEqual(pair['training_stream']['digest'],full['training_stream']['digest'])
            torch.testing.assert_close(pair['data_rng'],full['data_rng'],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
