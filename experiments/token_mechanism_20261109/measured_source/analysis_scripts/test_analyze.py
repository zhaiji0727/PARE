import unittest
from analyze import contrast,rank_metrics
class AnalysisChecks(unittest.TestCase):
 def test_exact_rank_metrics(self):
  r=rank_metrics([1,2,3,4],4);self.assertEqual(r['R@1'],25);self.assertEqual(r['R@5'],100);self.assertAlmostEqual(r['MRR@10'],100*(1+.5+1/3+.25)/4)
 def test_complete_seed_and_rank_requirement(self):
  with self.assertRaises(ValueError):contrast([1]*4,[1]*4)
  with self.assertRaises(ValueError):rank_metrics([1,0],2)
 def test_positive_stable_effect(self):
  r=contrast([2,2,2,2,2],[1]*5);self.assertTrue(r['predeclared_stronger_within_dataset_evidence'])
 def test_mixed_sign_not_success(self):
  r=contrast([5,5,5,5,-.1],[1]*5);self.assertFalse(r['predeclared_stronger_within_dataset_evidence']);self.assertLess(r['bonferroni_family95_interval_pp'][0],r['t95_interval_pp'][0])
 def test_mrr_decline_vetoes(self):
  self.assertFalse(contrast([2]*5,[-.01]*5)['predeclared_stronger_within_dataset_evidence'])
if __name__=='__main__':unittest.main()
