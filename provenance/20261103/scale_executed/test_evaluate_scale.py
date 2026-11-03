"""Numerical protocol checks for the independent scale evaluation adapter."""
import unittest

import torch

from evaluate_scale import full_gallery_head_ranks, validate_extra_tensors


def rank_of_gold(scores, candidates, gold):
    # Original evaluator's documented one-dimensional sorting convention.
    return [candidates[index] for index in torch.argsort(scores, descending=True).tolist()].index(gold) + 1


class ScaleEvaluationTests(unittest.TestCase):
    def test_known_full_gallery_ranks_and_scene_order_independence_without_ties(self):
        scores = torch.tensor([[4., 2., 3., 1.], [2., 3., 1., 4.]])
        for scenes in (scores, -scores, torch.zeros_like(scores)):
            self.assertEqual(full_gallery_head_ranks(scores, scenes, [0, 2], rank_of_gold), [1, 4])

    def test_ties_follow_scene_order_candidate_path(self):
        scores = torch.zeros((1, 4))
        scenes = torch.tensor([[1., 2., 3., 4.]])
        # Four equal PARE scores retain the reference CPU argsort convention:
        # the scene candidate order is [3,2,1,0], so target zero is fourth.
        self.assertEqual(full_gallery_head_ranks(scores, scenes, [0], rank_of_gold), [4])

    def test_missing_or_corrupt_physical_scalar_tensor_rejected(self):
        valid = {'ids': ['a', 'b'], 'L': torch.ones(2), 'c_star': torch.zeros(2)}
        validate_extra_tensors(valid)
        for wrong in (torch.ones(1), torch.ones(2, dtype=torch.float64), torch.tensor([1., float('nan')])):
            with self.assertRaises(ValueError):
                validate_extra_tensors({**valid, 'L': wrong})


if __name__ == '__main__':
    unittest.main(verbosity=2)
