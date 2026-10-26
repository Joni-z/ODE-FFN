import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from synthetic.initialization import copy_common_backbone
from synthetic.models import VelocityNet


class CommonBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'initial.pt'

    def model(self, kind):
        return VelocityNet(3, kind, width=16, depth=2, orders=1,
                           gate_hidden=8, gate_reference_hidden=16,
                           ffn_hidden=24).double()

    @staticmethod
    def snapshot(model):
        return dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
                    runtime_buffers={k:v.detach().clone() for k,v in model.named_buffers()})

    def save(self, state):
        torch.save(state, self.path)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for k in left:
                self.assert_tree_equal(left[k], right[k])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a,b in zip(left,right):
                self.assert_tree_equal(a,b)
        else:
            self.assertEqual(left,right)

    @staticmethod
    def rng():
        return dict(torch=torch.get_rng_state().clone(), python=random.getstate(),
                    numpy=np.random.get_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def test_pairs_surrounding_tensors_and_runtime_buffers_without_touching_ffn_or_rng(self):
        source, target = self.model('ode_vector'), self.model('swiglu')
        # Deliberately differ from platform-generated nonpersistent constants.
        with torch.no_grad():
            source.time.frequencies.add_(.125)
            source.output.weight.normal_()
        state = self.snapshot(source)
        digest = self.save(state)
        before, rng = self.snapshot(target), self.rng()
        result = copy_common_backbone(target, self.path, digest)
        after = self.snapshot(target)
        self.assert_tree_equal(rng, self.rng())
        self.assertGreater(result['max_before_difference'], 0)
        self.assertEqual(result['source_sha256'], digest)
        self.assertIn('time.frequencies', result['copied_runtime_buffer_keys'])
        self.assertNotIn('time.frequencies', result['copied_state_keys'])
        for k,v in after['model'].items():
            self.assert_tree_equal(v, before['model'][k] if '.ffn.' in k else state['model'][k])
        for k,v in after['runtime_buffers'].items():
            self.assert_tree_equal(v, before['runtime_buffers'][k] if '.ffn.' in k else state['runtime_buffers'][k])

    def test_rejects_hash_shape_and_missing_names_before_any_copy(self):
        source, target = self.model('ode_vector'), self.model('swiglu')
        source_state = self.snapshot(source)
        before, rng = self.snapshot(target), self.rng()
        cases = []
        bad = copy.deepcopy(source_state); bad['model']['input.weight'] = torch.zeros(1, 1)
        cases.append(('shape', bad, None))
        bad = copy.deepcopy(source_state); del bad['model']['blocks.1.condition.bias']
        cases.append(('missing state', bad, None))
        bad = copy.deepcopy(source_state); bad['model']['unexpected.weight'] = torch.zeros(1)
        cases.append(('extra state', bad, None))
        bad = copy.deepcopy(source_state); del bad['runtime_buffers']['time.frequencies']
        cases.append(('missing runtime buffer', bad, None))
        bad = copy.deepcopy(source_state); bad['runtime_buffers']['time.frequencies'] = torch.zeros(1)
        cases.append(('runtime buffer shape', bad, None))
        cases.append(('hash', source_state, '0'*64))
        for name, state, override in cases:
            with self.subTest(name=name):
                digest = self.save(state)
                with self.assertRaises(ValueError):
                    copy_common_backbone(target, self.path, override or digest)
                self.assert_tree_equal(before, self.snapshot(target))
                self.assert_tree_equal(rng, self.rng())

    def test_identity_copy_preserves_functions_gradients_and_three_adam_steps(self):
        reference = self.model('ode_vector')
        with torch.no_grad():
            reference.output.weight.normal_(std=.2)
        candidate = copy.deepcopy(reference)
        state = self.snapshot(reference)
        rng = self.rng()
        result = copy_common_backbone(candidate, self.path, self.save(state))
        self.assert_tree_equal(rng, self.rng())
        self.assertEqual(result['max_before_difference'], 0.)
        opt_a = torch.optim.AdamW(reference.parameters(), lr=3e-4, weight_decay=0)
        opt_b = torch.optim.AdamW(candidate.parameters(), lr=3e-4, weight_decay=0)
        for _ in range(3):
            x,t,y = torch.randn(17,3,dtype=torch.float64), torch.rand(17,1,dtype=torch.float64), torch.randn(17,3,dtype=torch.float64)
            opt_a.zero_grad(); opt_b.zero_grad()
            a,b = reference(x,t), candidate(x,t)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            ((a-y)**2).mean().backward(); ((b-y)**2).mean().backward()
            for p,q in zip(reference.parameters(),candidate.parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
            torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.)
            opt_a.step(); opt_b.step()
            self.assert_tree_equal(self.snapshot(reference), self.snapshot(candidate))
            self.assert_tree_equal(opt_a.state_dict(), opt_b.state_dict())


if __name__ == '__main__':
    unittest.main()
