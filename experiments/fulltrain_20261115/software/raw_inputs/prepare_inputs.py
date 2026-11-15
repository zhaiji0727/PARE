"""Plan or explicitly build new portable input caches from existing raw assets."""
import argparse
import json
import os
from pathlib import Path
from input_contract import ROOT, make_plan, require, write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path)
    parser.add_argument('--backbone', required=True, choices=('vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip'))
    parser.add_argument('--population', choices=('train', 'test'), default='train')
    parser.add_argument('--max-images', type=int, default=0, help='Positive prefix creates SMOKE_ONLY, never a training manifest')
    parser.add_argument('--execute', action='store_true', help='Explicitly construct and run pretrained encoders')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    args = parser.parse_args(argv)
    require(args.max_images >= 0, 'max-images must be nonnegative')
    if args.execute:
        require(args.device is not None, '--execute requires explicit --device')
    plan, ids, captions, _ = make_plan(args)
    if not args.execute:
        args.out.mkdir(parents=True, exist_ok=False)
        write(args.out / 'PLAN.json', plan)
        print(json.dumps({'status': plan['status'], 'plan': str(args.out / 'PLAN.json'),
                          'images': plan['images_to_encode'], 'models_constructed': False}))
        return
    if args.device == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        require(visible.isdigit(), 'Set CUDA_VISIBLE_DEVICES to one explicit GPU index')
    os.environ.update(RS_RAG_MODEL_ROOT=str(args.models.resolve()),
                      HF_HOME=str(args.models.resolve() / 'huggingface'),
                      HF_HUB_CACHE=str(args.models.resolve() / 'huggingface/hub'),
                      HUGGINGFACE_HUB_CACHE=str(args.models.resolve() / 'huggingface/hub'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if args.device == 'cuda':
        require(torch.cuda.device_count() == 1, 'Exactly one visible CUDA device required')
        torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    from raw_builder import build
    build(plan, ids, captions, args.out.resolve(), args.device)


if __name__ == '__main__':
    main()
