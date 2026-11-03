import copy
import itertools
import os
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.environ.get('FOLLOWUP_BASE_CODE', str(ROOT / 'base_code')), str(ROOT / 'code')]
from factorial_head import FactorialHead, Factors


def fixture():
    torch.manual_seed(4)
    n, d = 3, 16
    out = {'g': F.normalize(torch.randn(n, d), dim=-1), 'agg': torch.randn(n, d), 'nu': torch.rand(n)}
    for prefix, count in [('c', 2), ('f', 3)]:
        out[prefix + '_emb'] = F.normalize(torch.randn(n, count, d), dim=-1)
        out[prefix + '_mu'] = torch.rand(n, count)
        out[prefix + '_t'] = torch.rand(n, count)
        out[prefix + '_tf'] = torch.randn(n, count, 4)
        out[prefix + '_mask'] = torch.ones(n, count, dtype=torch.bool)
    out['c_mask'][1, 1] = False
    return F.normalize(torch.randn(d), dim=0), out


class FactorPathTests(unittest.TestCase):
    def test_gate_uses_exact_selected_aggregate_for_all_factors(self):
        query, docs = fixture()
        for flags in itertools.product([0, 1], repeat=5):
            factors = Factors(*flags)
            head = FactorialHead(16, factors)
            captured = []
            handle = head.gate.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
            result = head.fuse_batch(query, docs)
            handle.remove()
            expected = docs['agg'] if factors.B else (docs['g'] + docs['c_emb'].masked_fill(~docs['c_mask'].unsqueeze(-1), 0).sum(1)) / (1 + docs['c_mask'].sum(1, keepdim=True))
            self.assertTrue(torch.equal(captured[0][:, 16:], expected))
            self.assertTrue(torch.isfinite(result).all())

    def test_C_only_masks_coarse_fourth_descriptor_coordinate(self):
        query, docs = fixture()
        for c in [0, 1]:
            head = FactorialHead(16, Factors(C=c))
            original = head._augment_b
            captured = []
            def capture(emb, features):
                captured.append(features.detach().clone())
                return original(emb, features)
            head._augment_b = capture
            before = copy.deepcopy(docs)
            head.fuse_batch(query, docs)
            expected = docs['c_tf'].clone()
            if not c:
                expected[..., 3] = 0
            self.assertTrue(torch.equal(captured[0], expected))
            self.assertTrue(torch.equal(captured[1], docs['f_tf']))
            self.assertTrue(all(torch.equal(before[k], docs[k]) for k in docs))

    def test_A0_bypasses_fine_attention_but_retains_second_gate(self):
        query, docs = fixture()
        head = FactorialHead(16, Factors(A=0))
        coarse = F.normalize(torch.arange(1, 49).float().reshape(3, 16), dim=-1)
        head._attend_b = lambda q, emb, rho, mask: coarse
        with torch.no_grad():
            head.gate[-1].weight.zero_()
            head.gate[-1].bias.copy_(torch.logit(torch.tensor([.8, .7])))
        actual = head.fuse_batch(query, docs)
        expected = F.normalize(.8 * coarse + .7 * docs['g'], dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
        altered = copy.deepcopy(docs)
        for key in altered:
            if key.startswith('f_') and altered[key].dtype != torch.bool:
                altered[key].mul_(11)
        self.assertTrue(torch.equal(actual, head.fuse_batch(query, altered)))

    def test_C_has_no_effect_when_tokens_are_off_and_flags_cannot_mix(self):
        query, docs = fixture()
        for a, b, r in itertools.product([0, 1], repeat=3):
            left = FactorialHead(16, Factors(a, b, 0, r, 0))
            right = FactorialHead(16, Factors(a, b, 1, r, 0))
            right.load_state_dict(left.state_dict())
            self.assertTrue(torch.equal(left.fuse_batch(query, docs), right.fuse_batch(query, docs)))
        left.ablate = {'cfar'}
        with self.assertRaises(ValueError):
            left.fuse_batch(query, docs)
        with self.assertRaises(ValueError):
            Factors(A=2)


if __name__ == '__main__':
    unittest.main()
