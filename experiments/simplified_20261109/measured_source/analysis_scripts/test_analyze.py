import unittest
from analyze import contrast,rank_metrics,expected_keys,summary
class Checks(unittest.TestCase):
 def test_complete_fixed_matrix(self):
  keys=expected_keys();self.assertEqual(len(keys),105);self.assertEqual(len(set(keys)),105)
  self.assertEqual(sum(k[3]=='test' for k in keys),75)
 def test_rank_math(self):
  d=rank_metrics([1,2,11],20);self.assertAlmostEqual(d['R@10'],200/3);self.assertEqual(d['MRR@10'],50)
 def test_bad_ranks(self):
  for r in ([0],[True],[2.0],[3689],[]):
   with self.assertRaises(ValueError):rank_metrics(r)
 def test_fixed_pairs_required(self):
  for x in ([1]*4,[1]*6,[1,2,3,4,float('nan')]):
   with self.assertRaises(ValueError):summary(x)
 def test_stable_and_mixed_cases(self):
  self.assertTrue(contrast([1]*5,[.2]*5)['stable_positive_evidence'])
  self.assertFalse(contrast([1,2,3,4,-1],[.2]*5)['stable_positive_evidence'])
  self.assertFalse(contrast([1]*5,[-.2]*5)['stable_positive_evidence'])
 def test_family_interval_wider(self):
  d=contrast([1,2,3,4,5],[1]*5)
  self.assertLess(d['bonferroni_family95_interval'][0],d['t95_interval'][0])
  self.assertGreater(d['bonferroni_family95_interval'][1],d['t95_interval'][1])
if __name__=='__main__':unittest.main()
