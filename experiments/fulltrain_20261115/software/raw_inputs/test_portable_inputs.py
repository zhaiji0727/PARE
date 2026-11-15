import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runtime'))
import input_contract as contract
import torch
import numpy as np
from PIL import Image
import raw_builder
import v9_helpers
import saqre_phys
from text_helpers import encode_text, state_sha
from compare_smoke import compare


class FakeEncoder:
    def __init__(self, dimension):
        self.dimension = dimension
        self.image_batches = []
        self.text_batches = []

    def encode_image(self, batch):
        self.image_batches.append(len(batch))
        value = batch.flatten(1).mean(1) + len(batch) / 100
        result = torch.ones(len(batch), self.dimension)
        result[:, 0] = value
        return result

    def encode_text(self, batch):
        self.text_batches.append(len(batch))
        result = torch.ones(len(batch), self.dimension)
        result[:, 0] = batch[:, 0].float()
        return result


def preprocess(image):
    return torch.tensor(np.asarray(image.resize((8, 8))), dtype=torch.float32).permute(2, 0, 1) / 255


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Neither dependency is used by old.encode; avoid installing/downloading a model stack.
        with mock.patch.dict(sys.modules, {'open_clip': types.ModuleType('open_clip'),
                                          'tqdm': types.SimpleNamespace(tqdm=lambda x, **kwargs: x)}):
            import cache_primitives
        sys.modules['cache_primitives'] = cache_primitives

    def test_source_lock_and_unchanged_vendors(self):
        contract.verify_sources()
        provenance = json.loads((ROOT / 'SOURCE_PROVENANCE.json').read_text())
        for row in provenance['files']:
            if row.get('adaptation', '').startswith('none'):
                self.assertEqual(contract.sha(ROOT / row['destination']), row['source_sha256'])

    def test_cli_help_and_missing_inputs_do_not_import_torch(self):
        code = "import sys; sys.path.insert(0, sys.argv[1]); import prepare_inputs; assert 'torch' not in sys.modules; prepare_inputs.main(['--help'])"
        result = subprocess.run([sys.executable, '-c', code, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--data', result.stdout)

    def test_mixed_and_fixed_fine_policy_both_dimensions(self):
        # Only the module dependency is stubbed; the actual old encode function runs.
        for dimension in (512, 768):
            with self.subTest(dimension=dimension):
                array = np.zeros((64, 64, 3), dtype=np.uint8) + 20
                array[25:28, 30:33] = 255
                image = Image.fromarray(array)
                prim = saqre_phys.extract_primitives(image)
                model = FakeEncoder(dimension)
                cache = raw_builder.empty_cache(['synthetic.png'], dimension)
                record = raw_builder.fill_image(cache, 0, image, model, preprocess, 'cpu')
                self.assertEqual(model.image_batches, [1 + len(prim['regions']) + len(prim['scatterers']), 8])
                g = saqre_phys.to_intensity(image)
                expected = v9_helpers.legacy_boxes(g, saqre_phys.ca_cfar(g))
                self.assertEqual(record['legacy_boxes'], expected)
                self.assertEqual(int(cache['f_n'][0]), len(expected))
                nr, nf = int(cache['c_n'][0]), int(cache['f_n'][0])
                aggregate = torch.cat((cache['scene'], cache['c_emb'][0, :nr], cache['f_emb'][0, :nf])).mean(0)
                self.assertTrue(torch.equal(cache['agg'][0], aggregate))
                self.assertFalse(torch.cuda.is_initialized())

    def test_zero_detection_still_fixed_eight_call(self):
        image = Image.new('RGB', (64, 64), (0, 0, 0))
        model = FakeEncoder(512)
        cache = raw_builder.empty_cache(['zero.png'], 512)
        raw_builder.fill_image(cache, 0, image, model, preprocess, 'cpu')
        self.assertEqual(model.image_batches[-1], 8)
        self.assertEqual(int(cache['f_n'][0]), 0)
        self.assertEqual(int(torch.count_nonzero(cache['f_emb'])), 0)

    def test_text_loop_uses_old_256_batches(self):
        model = FakeEncoder(2)
        tokenizer = lambda texts: torch.tensor([[int(t)] for t in texts])
        result = encode_text([str(i) for i in range(513)], model, tokenizer, 'cpu')
        self.assertEqual(model.text_batches, [256, 256, 1])
        self.assertEqual(result.shape, (513, 2))
        self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(513)))

    def test_model_manifest_exact_assets_and_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models = root / 'relocated'; models.mkdir()
            weight = models / 'weights.bin'; weight.write_bytes(b'synthetic-weight')
            manifest = root / 'manifest.json'
            config = {'pretrained_manifest_tag': 'vitb32', 'expected_pretrained_sha256': contract.sha(weight),
                      'expected_external_sha256': None}
            manifest.write_text(json.dumps({'status': 'downloaded_and_hashed', 'assets': [
                {'tag': 'vitb32', 'path': 'models/weights.bin', 'sha256': contract.sha(weight), 'bytes': weight.stat().st_size}]}))
            result = contract.model_files(models, manifest, 'vitb32', config)
            self.assertEqual(result['pretrained']['path'], str(weight.resolve()))
            weight.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                contract.model_files(models, manifest, 'vitb32', config)

    def test_smoke_comparator_matches_by_id_and_exposes_differences(self):
        reference = raw_builder.empty_cache(['b', 'a'], 512)
        new = raw_builder.empty_cache(['a'], 512)
        self.assertTrue(compare(new, reference)['all_tensor_fields_exact'])
        new['scene'][0, 0] = 1
        result = compare(new, reference)
        self.assertFalse(result['all_tensor_fields_exact'])
        self.assertEqual(result['fields']['scene']['max_abs_difference'], 1)

    def test_smoke_comparator_rejects_missing_fields_and_duplicate_ids(self):
        reference = raw_builder.empty_cache(['a', 'b'], 512)
        new = raw_builder.empty_cache(['a'], 512)
        del new['f_emb']
        with self.assertRaisesRegex(ValueError, 'tensor fields'):
            compare(new, reference)
        new = raw_builder.empty_cache(['a', 'a'], 512)
        with self.assertRaisesRegex(ValueError, 'Duplicate new IDs'):
            compare(new, reference)

    def test_training_manifest_matches_actual_portable_validator(self):
        # Expanded synthetic views exercise official dimensions without large stored arrays.
        training = ROOT.parent / 'pare_portable_training_20261115'
        spec = importlib.util.spec_from_file_location('portable_training_contract', training / 'train_replay.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for name in ('cache_legacy_raster.pt', 'Caption_train.csv', 'training_text.pt'):
                (out / name).write_bytes(b'synthetic')
            cache = raw_builder.empty_cache(['one'], 512)
            for key, value in list(cache.items()):
                if torch.is_tensor(value):
                    cache[key] = value.expand(9181, *value.shape[1:])
            cache['ids'] = [f'synthetic_{i}' for i in range(9181)]
            signature = {'encoder': {'synthetic': True}}
            cache['provenance'] = {'signature': signature, 'signature_sha256': contract.objsha(signature)}
            vectors = torch.zeros(1, 512).expand(18461, 512)
            identity = {'training_ids_sha256': contract.objsha(cache['ids']), 'pair_order_sha256': 'synthetic',
                        'training_text_sha256': 'synthetic'}
            plan = {'backbone': 'vitb32', 'dimension': 512, 'smoke': False, 'source_lock_sha256': 'synthetic'}
            manifest = raw_builder.training_manifest(plan, cache, identity, vectors, out, state_sha)
            text = {'embeddings': vectors, 'backbone': 'vitb32', 'training_text_sha256': 'synthetic'}
            sys.path.insert(0, str(training / 'runtime'))
            try:
                result = module.validate_tensors(cache, text, manifest)
            finally:
                sys.path.remove(str(training / 'runtime'))
            self.assertEqual(result.shape, (18461, 512))
            self.assertEqual(manifest['encoder_signature_path'], ['encoder'])
            self.assertFalse(manifest['origin']['archived_byte_identity_claimed'])

    def synthetic_build(self, directory, fail_tail=False):
        root = Path(directory).resolve(); image_path = root / 'one.png'
        Image.new('RGB', (64, 64), (30, 30, 30)).save(image_path)
        caption_path = root / 'captions.csv'; caption_path.write_text('image,caption\none.png,3\n')
        plan = {'backbone': 'vitb32', 'population': 'train', 'dimension': 512,
                'selected_ids': ['one.png'], 'smoke': True, 'policy': {'synthetic': True},
                'source_lock_sha256': contract.sha(ROOT / 'SOURCE_LOCK.json'),
                'image_files': {'one.png': contract.binding(image_path)},
                'captions': contract.binding(caption_path), 'weights': {'external': None},
                'config': {'model': 'synthetic', 'pretrained': 'synthetic'}}
        encoder = {'synthetic': True}; model = FakeEncoder(512)
        tokenizer = lambda texts: torch.tensor([[int(t)] for t in texts])
        out = root / 'output'
        with mock.patch('raw_builder.setup_encoder', return_value=(model, preprocess, tokenizer, encoder, {})), \
             mock.patch('artifacts.extraction_signature', return_value=encoder), \
             mock.patch('raw_builder.recheck_plan', side_effect=ValueError('tail change') if fail_tail else None):
            if fail_tail:
                with self.assertRaisesRegex(ValueError, 'tail change'):
                    raw_builder.build(plan, ['one.png'], {'one.png': ['3']}, out, 'cpu')
            else:
                raw_builder.build(plan, ['one.png'], {'one.png': ['3']}, out, 'cpu')
        return out

    def test_synthetic_build_roundtrip_smoke_never_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.synthetic_build(directory)
            manifest = json.loads((out / 'SMOKE_INPUTS.json').read_text())
            self.assertEqual(manifest['status'], 'SMOKE_ONLY')
            self.assertFalse((out / 'INPUTS.json').exists())
            receipt = json.loads((out / 'BUILD_RECEIPT.json').read_text())
            self.assertFalse(receipt['reference_tensor_alignment_tested'])
            self.assertFalse(receipt['training_performed'])

    def test_tail_failure_cannot_emit_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.synthetic_build(directory, fail_tail=True)
            self.assertTrue((out / 'FAILED.json').exists())
            self.assertFalse((out / 'BUILD_RECEIPT.json').exists())
            self.assertFalse((out / 'SMOKE_INPUTS.json').exists())
            self.assertFalse((out / 'INPUTS.json').exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
