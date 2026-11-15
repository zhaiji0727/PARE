import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
import torch
from torch.nn import functional as F
from data_contract import objsha, query_sets
from pare_portable.spec import ModelSpec
from pare_portable.io import documents_from_cache
from pare_portable.api import score_queries
from scoring_contract import assemble_all_caption_text, report_both_query_modes
from text_helpers import state_sha
from evaluation_inputs import Reader, load_inputs, checkpoint_identity, sha, validate_fixed_queries
from evaluate import main


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    return {'path': str(path), 'sha256': sha(path)}


def fixture(root, backbone='vitb32', public=False, spec='path:100:full'):
    d = 768 if backbone == 'siglip' else 512
    ids = ['b', 'a', 'c']; n = len(ids)
    torch.manual_seed(11)
    encoder = {'model': 'synthetic-' + backbone, 'pretrained': 'synthetic',
               'pretrained_weight_sha256': '1' * 64, 'external_checkpoint_sha256': None}
    signature = {'schema': 'pare.portable-raw-cache.v1', 'origin': 'new_raw_image_reproduction',
                 'population': 'test', 'smoke': True, 'encoder': encoder, 'ids_sha256': objsha(ids)}
    cache = {'D': d, 'Mc': 6, 'Mf': 8, 'ids': ids, 'scene': F.normalize(torch.randn(n, d), dim=-1),
             'agg': torch.randn(n, d), 'nu': torch.rand(n), 'L': torch.ones(n), 'c_star': torch.ones(n),
             'provenance': {'schema_version': signature['schema'], 'signature': signature,
                            'signature_sha256': objsha(signature), 'ids_sha256': objsha(ids)}}
    for b, slots in [('c', 6), ('f', 8)]:
        cache[b + '_emb'] = F.normalize(torch.randn(n, slots, d), dim=-1)
        cache[b + '_mu'] = torch.rand(n, slots)
        cache[b + '_t'] = torch.rand(n, slots)
        cache[b + '_tf'] = torch.rand(n, slots, 4)
        cache[b + '_n'] = torch.tensor([slots, 1, 0])
    definitions = query_sets(ids, {'a': ['same', 'same'], 'b': ['other'], 'c': ['third']}, validate_official=False)
    text = {'first': F.normalize(torch.randn(3, d), dim=-1), 'extra': F.normalize(torch.randn(1, d), dim=-1)}
    torch.save(cache, root / 'cache.pt'); torch.save(text, root / 'text.pt')
    qb = write(root / 'queries.json', definitions)
    inputs = {'schema': 'pare.portable-retrieval-inputs.v1', 'status': 'SMOKE_ONLY',
              'backbone': backbone, 'dimension': d, 'encoder_signature': encoder,
              'cache': {'path': 'cache.pt', 'sha256': sha(root / 'cache.pt')},
              'query_text': {'path': 'text.pt', 'sha256': sha(root / 'text.pt')},
              'query_identity': {'path': 'queries.json', 'sha256': qb['sha256']}}
    ib = write(root / 'INPUTS.json', inputs)
    model_spec = ModelSpec.parse(spec)
    model = model_spec.build(d, torch.tensor([.1, .2, .3, .4]), torch.tensor([.4, .3, .2, .1]))
    variant = 'path_' + model_spec.abc + '__' + model_spec.control if model_spec.family == 'path' else model_spec.variant
    metadata = {'schema': 'pare.final-full-train.run.v1' if public else 'pare.portable-full-training-replay.v1',
                'backbone': backbone, 'dimension': d, 'variant': variant, 'seed': 1,
                'train_images': 9181, 'train_pairs': 31945, 'training_texts': 18461, 'epochs': 50, 'checkpoint_epochs': [20, 50],
                'initialization': 'fresh_from_seed', 'descriptor_mean_population': 'all_valid_regions_of_all_9181_training_images',
                'full_training_crop_mean_c': model.coarse_mean.tolist(),
                'full_training_crop_mean_f': model.fine_mean.tolist(), 'protocol_sha256': '2' * 64,
                'training_pair_order_sha256': '3' * 64, 'training_text_sha256': '4' * 64}
    train = {'schema': 'pare.portable-full-training-inputs.v1', 'status': 'BOUND',
             'backbone': backbone, 'dimension': d, 'encoder_signature': encoder,
             'training_ids_sha256': 'aebf2530ab350d94f3a325b514ba872ed1fa0e434ed9d7973c2c209a502b74f5',
             'pair_order_sha256': '3' * 64, 'training_text_sha256': '4' * 64}
    for key in ('cache', 'training_text', 'captions'):
        train[key] = {'path': 'not-needed-for-scoring', 'sha256': '5' * 64}
    tb = write(root / 'TRAIN_INPUTS.json', train)
    if not public:
        metadata['inputs_manifest_sha256'] = tb['sha256']
        metadata['input_files'] = {k: train[k] for k in ('cache', 'training_text', 'captions')}
    cp = {'state_dict': model.state_dict(), 'metadata': metadata}
    if not public:
        cp['history'] = [{'epoch': i, 'batches': 500} for i in range(1, 21)]
    torch.save(cp, root / 'checkpoint.pt')
    cb = {'path': str(root / 'checkpoint.pt'), 'sha256': sha(root / 'checkpoint.pt')}
    if public:
        row = {'backbone': backbone, 'variant': variant, 'seed': 1, 'epoch': 20, 'source_checkpoint_sha256': '6' * 64}
        audit = {'schema': 'pare.published-training-audit.v1', 'status': 'audit_passed',
                 'source_training_audit_sha256': '7' * 64, 'training_protocol_sha256': '2' * 64, 'checkpoints': [row]}
        audit_binding = write(root / 'PUBLIC_AUDIT.json', audit)
        sidecar = {'schema': 'pare.published-checkpoint-binding.v1', 'status': 'projected_from_audited_checkpoint',
                   **row, 'checkpoint_sha256': cb['sha256'], 'state_dict_sha256': state_sha(cp['state_dict']),
                   'training_protocol_sha256': '2' * 64, 'source_training_audit_sha256': '7' * 64,
                   'encoder_signature': encoder, 'training_population': {'images': 9181, 'pairs': 31945, 'texts': 18461},
                   'training_audit': {'path': 'PUBLIC_AUDIT.json', 'sha256': audit_binding['sha256']}}
        pb = write(root / 'PUBLISHED_BINDING.json', sidecar)
    else:
        pb = None
    return dict(inputs=ib, checkpoint=cb, training=tb, publication=pb, model=model, cache=cache,
                text=text, queries=definitions, spec=spec, backbone=backbone, checkpoint_object=cp)


class Tests(unittest.TestCase):
    def test_full_cli_both_dimensions_and_both_provenance_paths(self):
        for bone, public in [('vitb32', False), ('siglip', True)]:
            with self.subTest(backbone=bone, public=public), tempfile.TemporaryDirectory() as td:
                root = Path(td); f = fixture(root, bone, public)
                provenance = f['publication'] if public else f['training']
                name = '--checkpoint-binding' if public else '--training-inputs'
                args = ['--checkpoint', f['checkpoint']['path'], '--checkpoint-sha256', f['checkpoint']['sha256'],
                        '--inputs', f['inputs']['path'], '--inputs-sha256', f['inputs']['sha256'],
                        name, provenance['path'], name + '-sha256', provenance['sha256'],
                        '--model', f['spec'], '--backbone', bone, '--device', 'cpu', '--out', str(root / 'out'), '--smoke']
                main(args)
                result = json.loads((root / 'out/RESULT.json').read_text())
                self.assertFalse(result['eligible_for_formal_paper_tables'])
                self.assertEqual(result['scope'], 'smoke_only')
                self.assertEqual(result['first-caption']['source_all_caption_rows'], [0, 2, 3])
                scores = torch.load(root / 'out/scores.pt', weights_only=True)['scores']
                all_text = assemble_all_caption_text(f['text']['first'], f['text']['extra'], f['queries']['all_caption_row_map'])
                expected = score_queries(f['model'].eval(), all_text, documents_from_cache(f['cache'], f['cache']['ids']), query_batch_size=8)
                self.assertTrue(torch.equal(scores, expected))
                self.assertFalse(torch.cuda.is_initialized())

    def test_exact_ties_and_nonprefix_first(self):
        with tempfile.TemporaryDirectory() as td:
            f = fixture(Path(td))
            scores = torch.ones(4, 3)
            report = report_both_query_modes(scores, f['queries']['all'], f['queries']['first'], f['cache']['ids'])
            self.assertEqual(report['all-captions']['ranks'], [2, 2, 1, 3])
            self.assertEqual(report['first-caption']['ranks'], [2, 1, 3])
            self.assertEqual(report['first-caption']['source_all_caption_rows'], [0, 2, 3])

    def test_smoke_cannot_be_opened_as_official_population(self):
        with tempfile.TemporaryDirectory() as td:
            f = fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'status/schema'):
                load_inputs(Reader(), f['inputs'], 'vitb32', False)

    def test_wrong_same_dimension_backbone_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f = fixture(Path(td))
            with self.assertRaisesRegex(ValueError, 'backbone'):
                load_inputs(Reader(), f['inputs'], 'remoteclip', True)
            with self.assertRaisesRegex(ValueError, 'backbone'):
                checkpoint_identity(Reader(), f['checkpoint_object'], f['checkpoint'], 'remoteclip', f['spec'],
                                    f['cache']['provenance']['signature']['encoder'], f['training'])

    def test_query_order_or_caption_identity_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); f = fixture(root)
            q = json.loads((root / 'queries.json').read_text()); q['all_caption_row_map'] = list(reversed(q['all_caption_row_map']))
            qb = write(root / 'queries.json', q)
            i = json.loads((root / 'INPUTS.json').read_text()); i['query_identity']['sha256'] = qb['sha256']
            ib = write(root / 'INPUTS.json', i)
            with self.assertRaisesRegex(ValueError, 'query identities/order'):
                load_inputs(Reader(), ib, 'vitb32', True)

    def test_public_sidecar_binds_state_and_audit_membership(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); f = fixture(root, public=True)
            sidecar = json.loads((root / 'PUBLISHED_BINDING.json').read_text())
            sidecar['state_dict_sha256'] = '0' * 64
            pb = write(root / 'PUBLISHED_BINDING.json', sidecar)
            with self.assertRaisesRegex(ValueError, 'checkpoint/state'):
                checkpoint_identity(Reader(), f['checkpoint_object'], f['checkpoint'], 'vitb32', f['spec'],
                                    f['cache']['provenance']['signature']['encoder'], published_binding=pb)

    def test_input_encoder_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f = fixture(Path(td)); encoder = copy.deepcopy(f['cache']['provenance']['signature']['encoder'])
            encoder['pretrained_weight_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'encoder'):
                checkpoint_identity(Reader(), f['checkpoint_object'], f['checkpoint'], 'vitb32', f['spec'], encoder, f['training'])

    def test_same_count_rewritten_caption_rejected_by_fixed_protocol(self):
        ids = ['a', 'b']
        original = query_sets(ids, {'a': ['first', 'extra'], 'b': ['second']}, validate_official=False)
        digests = {}
        for name in ('first', 'extra', 'all'):
            digests[name + '_text_sha256'] = objsha([r['caption_sha256'] for r in original[name]])
            digests[name + '_query_ids_sha256'] = objsha([r['query_id'] for r in original[name]])
        digests['all_caption_row_map_sha256'] = objsha(original['all_caption_row_map'])
        fixed = {'digests': digests}
        validate_fixed_queries(original, fixed)
        changed = query_sets(ids, {'a': ['first', 'rewritten'], 'b': ['second']}, validate_official=False)
        self.assertEqual([len(changed[n]) for n in ('first', 'extra', 'all')], [2, 1, 3])
        self.assertEqual([r['query_id'] for r in changed['all']], [r['query_id'] for r in original['all']])
        with self.assertRaisesRegex(ValueError, 'Caption content or query order'):
            validate_fixed_queries(changed, fixed)

    def test_help_does_not_import_torch(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import evaluate;assert 'torch' not in sys.modules;evaluate.main(['--help'])"
        result = subprocess.run([sys.executable, '-c', code, str(ROOT)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
