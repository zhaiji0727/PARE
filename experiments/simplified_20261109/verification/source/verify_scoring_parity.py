#!/usr/bin/env python3
"""Prepared scoring-only check: audited B32/SigLIP, seed0, all three variants.

Run only after the main study releases a GPU. Never trains or changes a head.
The optional projected-head root exercises the metadata-redacted weights too.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'repo', 'portable', 'models', 'hf-home', 'independent-audit', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--projected-heads', type=Path)
    p.add_argument('--measured-head-manifest', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a new parity-check directory')
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not selected or ',' in selected:
        raise ValueError('Select one available physical GPU with CUDA_VISIBLE_DEVICES')
    import torch
    if not torch.cuda.is_available():
        raise ValueError('This scoring parity check requires CUDA, never CPU tie ordering')
    audit = json.loads(a.independent_audit.read_text())
    if not (audit.get('status') == 'pass' and audit.get('heads') == 75 and audit.get('evaluations') == 105):
        raise ValueError('Completed independent study audit required')
    req = json.loads((a.portable/'SOURCE_REQUIREMENTS.json').read_text())
    if any(audit[k] != req['measured_study'][k] for k in ('freeze_sha256', 'protocol_sha256')):
        raise ValueError('Audit and portable study bindings differ')
    bound = json.loads((a.study/'INPUTS_BOUND.json').read_text())
    env = {**os.environ, 'RS_RAG_ROOT': str(a.repo.resolve()), 'RS_RAG_MODEL_ROOT': str(a.models.resolve()),
           'HF_HOME': str(a.hf_home.resolve()), 'HF_HUB_CACHE': str(a.hf_home.resolve()/'hub'),
           'HUGGINGFACE_HUB_CACHE': str(a.hf_home.resolve()/'hub'), 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PARE_DEVICE': 'cuda:0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    a.out.mkdir(parents=True)
    reports = []
    for tag in ('vitb32', 'siglip'):
        b = req['backbones'][tag]
        cache = Path(bound['backbones'][tag]['files']['test_cache']['path'])
        captions = Path(bound['common']['test_captions']['path'])
        if sha(cache) != bound['backbones'][tag]['files']['test_cache']['sha256'] or sha(captions) != bound['common']['test_captions']['sha256']:
            raise ValueError('Original evaluation input changed')
        for variant in ('full', 'strict', 'strict_token_off'):
            relative = Path(tag)/'seed_0'/variant
            original = a.study/'results'/relative
            for name in ('head.pt', 'test.json', 'test_scores.pt'):
                if sha(original/name) != audit['artifact_sha256'][str(Path('results')/relative/name)]:
                    raise ValueError('Audited original changed: ' + str(relative/name))
            expected = json.loads((original/'test.json').read_text())
            matrix = torch.load(original/'test_scores.pt', map_location='cpu', weights_only=True)['scores']
            modes = [('original', original/'head.pt')]
            if a.projected_heads:
                modes.append(('projected', a.projected_heads/relative/'head.pt'))
            for mode, head in modes:
                stem = tag+'_'+variant+'_'+mode
                out = a.out/(stem+'.json')
                scores = a.out/(stem+'_scores.pt')
                argv = [sys.executable, str(a.portable/'evaluate.py'), '--repo', str(a.repo), '--cache', str(cache),
                        '--reference-cache', str(cache), '--captions', str(captions), '--head', str(head), '--out', str(out),
                        '--scores-out', str(scores), '--model', b['model'], '--pretrained', b['pretrained'],
                        '--backbone', tag, '--seed', '0', '--variant', variant, '--label', 'test', '--factor', '1', '--device', 'cuda:0']
                argv += ['--measured-head-manifest', str(a.measured_head_manifest)]
                with (a.out/(stem+'.log')).open('x') as stream:
                    subprocess.run(argv, cwd=a.repo, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                result = json.loads(out.read_text())
                produced = torch.load(scores, map_location='cpu', weights_only=True)['scores']
                for key in ('query_ids', 'query_image_ids', 'query_caption_sha256', 'gallery_ids', 'gold_gallery_index', 'ranks'):
                    if result[key] != expected[key]:
                        raise ValueError('Portable scoring identity/rank mismatch: ' + stem+'/'+key)
                if produced.shape != matrix.shape or produced.dtype != matrix.dtype or not torch.isfinite(produced).all():
                    raise ValueError('Score tensor shape/dtype/finite mismatch')
                reports.append({'backbone': tag, 'variant': variant, 'seed': 0, 'checkpoint_mode': mode,
                                'n_queries': 3688, 'n_gallery': 3688, 'all_ranks_identical': True,
                                'full_score_matrix_identical': bool(torch.equal(matrix, produced)),
                                'max_abs_score_difference': float((matrix-produced).abs().max()),
                                'original_head_sha256': sha(original/'head.pt'), 'evaluated_head_sha256': sha(head),
                                'result_sha256': sha(out), 'score_artifact_sha256': sha(scores)})
    expected_count = 12 if a.projected_heads else 6
    if len(reports) != expected_count:
        raise ValueError('Incomplete parity grid')
    report = {'status': 'all_scoring_rank_vectors_match', 'training_performed': False,
              'backbones': ['vitb32', 'siglip'], 'seeds': [0], 'variants': ['full', 'strict', 'strict_token_off'],
              'projection_tested': bool(a.projected_heads), 'checks': reports,
              'independent_audit_sha256': sha(a.independent_audit), 'parity_source_sha256': sha(__file__),
              'portable_source_sha256': {p.name: sha(p) for p in sorted(a.portable.glob('*.py'))},
              'requirements_sha256': sha(a.portable/'SOURCE_REQUIREMENTS.json'),
              'limits': 'Scoring parity for these six audited seed0 heads only; no retraining equivalence or other seeds inferred.'}
    (a.out/'PARITY_VERIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'checks': len(reports)}))


if __name__ == '__main__':
    main()
