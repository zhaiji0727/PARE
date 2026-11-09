#!/usr/bin/env python3
"""New path-portable full-gallery adapter using the published numerical kernel.

No historical anchors are required. Output is a NEW reproduction, not an
archived result; caption bodies and local paths are excluded from result JSON.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from checkpoint_contract import validate_original_trainer, validate_checkpoint, parameter_records, validate_parameter_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_identity(args, checkpoint, signature, requirements):
    validate_original_trainer(args, checkpoint, signature, requirements)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'cache', 'reference-cache', 'captions', 'head', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--remoteclip', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, required=True, choices=range(5))
    p.add_argument('--backbone', required=True)
    p.add_argument('--variant', required=True, choices=('full', 'strict', 'strict_token_off'))
    p.add_argument('--label', required=True, choices=('test', 'spacing_1p5', 'spacing_2'))
    p.add_argument('--factor', required=True, choices=('1', '1.5', '2'))
    p.add_argument('--scores-out', type=Path, help='Optional score tensor for independent parity checks; never required for normal reproduction')
    p.add_argument('--measured-head-manifest', type=Path, help='Required parameter identities for actual measured heads; defaults to repo/checkpoints/simplified_20261109/HEAD_PROJECTION_MANIFEST.json')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.scores_out and a.scores_out.exists():
        raise FileExistsError(a.scores_out)
    if {'test': '1', 'spacing_1p5': '1.5', 'spacing_2': '2'}[a.label] != a.factor:
        raise ValueError('Scale label/factor mismatch')
    repo = a.repo.resolve()
    from run import validate_repo
    requirements = validate_repo(repo)
    sys.path[:0] = [str(repo/'code'), str(repo/'tools/evaluation'), str(repo/'tools/scale')]
    import torch
    import torch.nn.functional as F
    import open_clip
    from artifacts import validate_cache, validate_head
    from train_saqre import load_captions, move_docs
    from batch_scoring import score_matrix
    from saqre_phys import SAQREHead
    from runtime import create_backbone
    from retrieval_queries import select_queries
    from evaluate_multik import score_head_matrix, rank_of_gold, full_gallery_ranks, metrics, validate_tensor_layout
    from stress_cache import validate_scale_cache
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    cache = torch.load(a.cache, map_location='cpu', weights_only=True)
    if a.factor == '1':
        signature = validate_cache(cache, a.model, a.pretrained, a.remoteclip)
        view = cache
    else:
        reference = torch.load(a.reference_cache, map_location='cpu', weights_only=True)
        view = validate_scale_cache(cache, reference, sha(a.reference_cache), a.factor, a.model, a.pretrained, a.remoteclip)
        signature = view['provenance']['signature']
    validate_tensor_layout(cache)
    checkpoint = torch.load(a.head, map_location='cpu', weights_only=True)
    manifest_path = a.measured_head_manifest or repo/'checkpoints/simplified_20261109/HEAD_PROJECTION_MANIFEST.json'
    def measured_parameter_check(state):
        manifest = json.loads(manifest_path.read_text())
        validate_parameter_manifest(a, parameter_records(state, torch), requirements, manifest)
    checkpoint_origin = validate_checkpoint(a, checkpoint, view, requirements, validate_head, measured_parameter_check)
    flags = {'full': [], 'strict': ['cfar_strict'], 'strict_token_off': ['cfar_strict', 'ty']}[a.variant]
    expected_ids = json.loads((repo/'data/splits_content_disjoint_20261007/test_ids.json').read_text())
    if cache['ids'] != expected_ids or sha(a.captions) != requirements['clean_caption_sha256']['Caption_test.csv']:
        raise ValueError('Standalone evaluator requires the fixed test IDs/order and original caption bytes')
    records, _ = select_queries(load_captions(a.captions), cache['ids'], 'first-caption', 0)
    if len(records) != 3688 or len(cache['ids']) != 3688:
        raise ValueError('Exactly 3688 first-caption queries/gallery images required')
    model, _, _ = create_backbone(a.model, a.pretrained, a.remoteclip)
    model = model.to(a.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(a.model)
    with torch.no_grad():
        queries = torch.cat([F.normalize(model.encode_text(tokenizer([r['caption'] for r in records[i:i+256]]).to(a.device)), dim=-1).cpu()
                             for i in range(0, len(records), 256)])
    del model
    head = SAQREHead(cache['D']).to(a.device)
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    head.ablate = set(flags)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    gold = [r['gold_gallery_row'] for r in records]
    with torch.no_grad():
        scene = queries.to(a.device) @ F.normalize(cache['scene'].to(a.device), dim=-1).T
        scores = score_head_matrix(head, queries, cache, move_docs, score_matrix, a.device, 32, 512).to(a.device)
        order = torch.argsort(scene, dim=1, descending=True)
        ranks = [rank_of_gold(scores[i, order[i]], order[i].tolist(), g) for i, g in enumerate(gold)]
        scene_ranks = full_gallery_ranks(scene, gold)
    result = {
        'schema': 'pare.portable-full-gallery-reproduction.v1',
        'attribution': 'new portable reproduction; not measured by archived study runner',
        'checkpoint_training_origin': checkpoint_origin,
        'checkpoint_schema': checkpoint.get('schema', 'original_train_saqre'),
        'measured_parameter_manifest_sha256': sha(manifest_path) if checkpoint_origin == 'actual_measured_simplified_study' else None,
        'backbone': a.backbone, 'variant': a.variant, 'seed': a.seed, 'evaluation_label': a.label,
        'n_queries': 3688, 'n_gallery': 3688, 'ranks': ranks, 'metrics_percent': metrics(ranks),
        'query_ids': [r['query_id'] for r in records], 'query_image_ids': [r['image_id'] for r in records],
        'query_caption_sha256': [r['caption_sha256'] for r in records], 'gallery_ids': cache['ids'], 'gold_gallery_index': gold,
        'scene_ranks': scene_ranks, 'scene_metrics_percent': metrics(scene_ranks),
        'checkpoint_sha256': sha(a.head), 'evaluation_cache_sha256': sha(a.cache), 'captions_sha256': sha(a.captions),
        'adapter_sha256': sha(__file__), 'kernel_sha256': sha(repo/'tools/evaluation/evaluate_multik.py'),
        'source_requirements_sha256': sha(Path(__file__).with_name('SOURCE_REQUIREMENTS.json')),
        'ranking': 'full gallery; scene 2D argsort then per-query 1D head argsort in scene order',
        'physical_GSD_claim': False, 'original_test_previously_seen': True,
    }
    if a.scores_out:
        a.scores_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'scores': scores.detach().cpu(), 'scene_scores': scene.detach().cpu(),
                    'query_ids': result['query_ids'], 'gallery_ids': result['gallery_ids'],
                    'checkpoint_sha256': result['checkpoint_sha256']}, a.scores_out)
        result['scores_artifact_sha256'] = sha(a.scores_out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
