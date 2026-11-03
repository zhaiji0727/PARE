import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get('PARE_ORIGINAL_ROOT', ROOT.parent / 'original'))
sys.path.insert(0, str(ROOT / 'scripts'))
from protocol_adapter import (ORIGINAL_PROTOCOL_SHA256, check_cache_contract,
                              check_scientific_identity, load_original,
                              make_reproduction_protocol, sha, validate_reproduction_protocol)


class PortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_original(ORIGINAL)

    def make(self):
        return make_reproduction_protocol(self.original, 'a' * 64, 'b' * 64,
                                          '2026-11-03T04:00:00+00:00')

    def test_original_protocol_is_exact_and_accepted(self):
        plan = validate_reproduction_protocol(ORIGINAL / 'plans/protocol.json',
                                               ORIGINAL_PROTOCOL_SHA256, ORIGINAL)
        self.assertEqual(plan, self.original)

    def test_only_byte_bindings_and_new_metadata_change(self):
        plan = self.make()
        self.assertEqual(plan['fixed_utc'], self.original['fixed_utc'])
        check_scientific_identity(plan, self.original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            path.write_text(json.dumps(plan, indent=2) + '\n')
            shutil.copyfile(ORIGINAL / 'plans/protocol.md', path.with_suffix('.md'))
            self.assertEqual(validate_reproduction_protocol(path, sha(path), ORIGINAL), plan)

    def test_rejects_scientific_and_reporting_amendments(self):
        mutations = [
            lambda p: p['training'].__setitem__('lr', .003),
            lambda p: p['training'].__setitem__('epochs', 5),
            lambda p: p['seeds'].__setitem__(0, 5),
            lambda p: p['configurations'].pop(),
            lambda p: p['paired_contrasts'][0]['terms'][0].__setitem__('coefficient', 2),
            lambda p: p['input_bindings'].__setitem__('clean_training_captions_sha256', 'c' * 64),
            lambda p: p['input_bindings']['split_file_sha256'].__setitem__('fit_ids.json', 'c' * 64),
            lambda p: p['backbone'].__setitem__('pretrained_weight_sha256', 'c' * 64),
            lambda p: p['evidence_status'].__setitem__('new_blinded_confirmatory_test', True),
            lambda p: p.__setitem__('fixed_utc', '2026-11-03T04:00:00+00:00'),
        ]
        for mutate in mutations:
            plan = self.make()
            mutate(plan)
            with self.assertRaises(ValueError):
                check_scientific_identity(plan, self.original)

    def test_rejects_forged_parent_or_extra_metadata(self):
        for key, value in [('parent_protocol_sha256', 'c' * 64),
                           ('scientific_design_unchanged', False),
                           ('unreviewed_override', True), ('created_utc', 'undated')]:
            plan = self.make()
            plan['reproduction'][key] = value
            with self.assertRaises(ValueError):
                check_scientific_identity(plan, self.original)

    def test_rejects_wrong_expected_hash_and_modified_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            path.write_text(json.dumps(self.make()))
            shutil.copyfile(ORIGINAL / 'plans/protocol.md', path.with_suffix('.md'))
            with self.assertRaises(ValueError):
                validate_reproduction_protocol(path, '0' * 64, ORIGINAL)
            path.with_suffix('.md').write_text('Different protocol')
            with self.assertRaises(ValueError):
                validate_reproduction_protocol(path, sha(path), ORIGINAL)

    def test_rejects_unlabelled_reformatted_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            path.write_text(json.dumps(self.original))
            shutil.copyfile(ORIGINAL / 'plans/protocol.md', path.with_suffix('.md'))
            with self.assertRaises(ValueError):
                validate_reproduction_protocol(path, sha(path), ORIGINAL)

    def test_cache_contract_keeps_fixed_population_and_backbone(self):
        ids = [str(i) for i in range(9181)]
        cache = {'D': 512, 'Mc': 6, 'Mf': 8, 'ids': ids}
        signature = {'primitive_mode': 'physics', 'model': self.original['backbone']['model'],
                     'pretrained': self.original['backbone']['pretrained'],
                     'pretrained_weight_sha256': self.original['backbone']['pretrained_weight_sha256'],
                     'external_checkpoint_sha256': None}
        check_cache_contract(cache, signature, self.original, ids)
        for key, value in [('D', 768), ('Mc', 5), ('Mf', 7), ('ids', ids[:-1] + [ids[0]])]:
            changed = dict(cache, **{key: value})
            with self.assertRaises(ValueError):
                check_cache_contract(changed, signature, self.original, ids)
        for key, value in [('primitive_mode', 'uniform_grid'), ('model', 'ViT-B-16'),
                           ('pretrained_weight_sha256', 'c' * 64), ('external_checkpoint_sha256', 'c' * 64)]:
            changed = dict(signature, **{key: value})
            with self.assertRaises(ValueError):
                check_cache_contract(cache, changed, self.original, ids)

    def test_factor_adapter_and_math_functions_are_unchanged(self):
        self.assertEqual(sha(ROOT / 'code/factorial_head.py'), sha(ORIGINAL / 'code/factorial_head.py'))
        original = ast.parse((ORIGINAL / 'scripts/run_factorial_fitval.py').read_text())
        portable = ast.parse((ROOT / 'scripts/run_factorial_fitval.py').read_text())
        for function in ['train_one', 'validate_one', 'state_sha']:
            before = next(x for x in original.body if isinstance(x, ast.FunctionDef) and x.name == function)
            after = next(x for x in portable.body if isinstance(x, ast.FunctionDef) and x.name == function)
            self.assertEqual(ast.dump(before), ast.dump(after), function)


if __name__ == '__main__':
    unittest.main()
