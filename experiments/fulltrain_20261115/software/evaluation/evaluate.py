"""Evaluate a reader's newly rebuilt test inputs; never writes formal paper results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from evaluation_inputs import Reader, checkpoint_identity, load_inputs, require, sha, source_lock


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'inputs'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument('--training-inputs', type=Path)
    provenance.add_argument('--checkpoint-binding', type=Path)
    parser.add_argument('--training-inputs-sha256')
    parser.add_argument('--checkpoint-binding-sha256')
    parser.add_argument('--model', required=True, help='Explicit portable model, e.g. path:100:full or joint:adapter')
    parser.add_argument('--backbone', required=True, choices=('vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip'))
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true', help='Accept only an explicitly SMOKE_ONLY test input manifest')
    args = parser.parse_args(argv)
    require(not args.out.exists(), 'Output must be new; no overwrite/resume')
    reader = Reader(); lock_sha = source_lock(reader)
    def bound(name):
        path = getattr(args, name.replace('-', '_'))
        if path is None:
            return None
        result = {'path': str(path.resolve()), 'sha256': getattr(args, name.replace('-', '_') + '_sha256')}
        reader.bind(result)
        return result
    checkpoint_binding, inputs_binding = bound('checkpoint'), bound('inputs')
    training_inputs, publication_binding = bound('training-inputs'), bound('checkpoint-binding')
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='3', MKL_NUM_THREADS='3',
                      OPENBLAS_NUM_THREADS='3', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.device == 'cuda':
        require(os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit(), 'Select one GPU with CUDA_VISIBLE_DEVICES')
    import torch
    from pare_portable.api import load_model
    from pare_portable.io import load_tensor_artifact
    from evaluate_core import evaluate_model
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    manifest, cache, text, queries = load_inputs(reader, inputs_binding, args.backbone, args.smoke)
    checkpoint = load_tensor_artifact(args.checkpoint, args.checkpoint_sha256)
    identity = checkpoint_identity(reader, checkpoint, checkpoint_binding, args.backbone, args.model,
                                   manifest['encoder_signature'], training_inputs, publication_binding)
    model, metadata = load_model(args.checkpoint, args.checkpoint_sha256, args.model,
                                 expected_backbone=args.backbone, device='cpu')
    del checkpoint
    reader.finish()
    if args.device == 'cuda':
        require(torch.cuda.device_count() == 1, 'Exactly one CUDA device must be visible')
    model.to(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        scores, report = evaluate_model(model, cache, text['first'], text['extra'], queries,
                                        cache['ids'], query_batch_size=8, official=not args.smoke)
        with (args.out / 'scores.pt').open('xb') as stream:
            torch.save({'scores': scores, 'query_ids': [q['query_id'] for q in queries['all']],
                        'gallery_ids': cache['ids'], 'gold': [q['gold_gallery_row'] for q in queries['all']],
                        'checkpoint_sha256': args.checkpoint_sha256, 'input_manifest_sha256': args.inputs_sha256}, stream)
        rank_keys = ('all-captions', 'first-caption', 'tie_rule', 'gallery_size')
        write(args.out / 'RANKS.json', {key: report[key] for key in rank_keys})
        report.update(schema='pare.reader-reproduction-evaluation.v1',
                      scope='smoke_only' if args.smoke else 'full_benchmark_reader_reproduction',
                      eligible_for_formal_paper_tables=False, original_experiment_rerun=False,
                      checkpoint_identity=identity, checkpoint_sha256=args.checkpoint_sha256,
                      input_manifest=inputs_binding, source_lock_sha256=lock_sha,
                      source_and_input_sha256=dict(reader.files), torch=str(torch.__version__),
                      gpu_name=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                      started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
                      scores_sha256=sha(args.out / 'scores.pt'))
        reader.finish()
        write(args.out / 'RESULT.json', report)
        write(args.out / 'complete.json', {'schema': 'pare.reader-reproduction-completion.v1', 'status': 'completed',
              'scope': report['scope'], 'eligible_for_formal_paper_tables': False,
              'artifacts': {p.name: sha(p) for p in args.out.iterdir() if p.is_file()}})
        print(json.dumps({'status': 'completed', 'scope': report['scope'], 'out': str(args.out)}))
    except BaseException as error:
        write(args.out / 'failed.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
