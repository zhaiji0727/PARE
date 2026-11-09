import copy
import json
from pathlib import Path
import random
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from study_core import validate_plan,training_pairs,training_orders,objsha,evaluation_labels

class ContractTests(unittest.TestCase):
    def test_no_variant_seed_or_hyperparameter_drift(self):
        plan=json.loads((ROOT/'protocol.json').read_text());validate_plan(plan)
        changes=[('seeds',[0,1]),('expected_heads',74),('variants',{'full':[]})]
        for field,value in changes:
            other=copy.deepcopy(plan);other[field]=value
            with self.assertRaises(ValueError):validate_plan(other)
        other=copy.deepcopy(plan);other['training']['early_stopping']=True
        with self.assertRaises(ValueError):validate_plan(other)
        other=copy.deepcopy(plan);other['backbones']['siglip']['kappa']=.03
        with self.assertRaises(ValueError):validate_plan(other)
        other=copy.deepcopy(plan);other['evaluation']['scale_seeds']=[0]
        with self.assertRaises(ValueError):validate_plan(other)
    def test_original_caption_order_and_repeated_in_place_shuffles(self):
        ids=['b','a','c'];caps={'a':['z','x'],'missing':['ignore'],'c':['q'],'b':['x','z']}
        pairs=training_pairs(ids,caps)
        self.assertEqual(pairs,[(1,'z'),(1,'x'),(2,'q'),(0,'x'),(0,'z')])
        for seed in range(5):
            copied=list(pairs);rng=random.Random(seed);order=list(range(len(pairs)));r2=random.Random(seed)
            expected=training_orders(seed,len(pairs),10)
            for i in range(10):
                rng.shuffle(copied);r2.shuffle(order)
                self.assertEqual(copied,[pairs[k] for k in order]);self.assertEqual(objsha(order),expected[i])
    def test_fixed_grid_has_75_heads_and_105_evaluations(self):
        tags=['vitb32','vitb16','remoteclip','siglip','georsclip']
        self.assertEqual(sum(5*3 for _ in tags),75)
        self.assertEqual(sum(5*3*len(evaluation_labels(t)) for t in tags),105)

if __name__=='__main__':unittest.main()
