#!/usr/bin/env python3
"""New portable reproduction entry; NOT the executor of archived measurements.

Calls the repository's original train_saqre.py for all 75 fixed heads.
Does not require a historical head, score matrix, private path, or anchor.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BACKBONES = {
    'vitb32': ('ViT-B-32', 'laion2b_s34b_b79k', .03, None),
    'vitb16': ('ViT-B-16', 'laion2b_s34b_b88k', .05, None),
    'remoteclip': ('ViT-B-32', 'laion2b_s34b_b79k', .05, 'external_checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt'),
    'siglip': ('ViT-B-16-SigLIP', 'webli', .07, None),
    'georsclip': ('ViT-B-32', 'openai', .05, 'external_checkpoints/georsclip/ckpt/RS5M_ViT-B-32.pt'),
}
VARIANTS = {'full': '', 'strict': 'cfar_strict', 'strict_token_off': 'cfar_strict,ty'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(repo, data, models, out, python=sys.executable, device='cuda:0'):
    repo, data, models, out = map(Path, (repo, data, models, out))
    jobs = []
    clean = data/'ret_clean_20261007'
    def add(stage, name, argv, outputs):
        jobs.append({'stage': stage, 'name': name, 'argv': [str(v) for v in argv], 'outputs': [str(v) for v in outputs]})
    for tag, (model, pretrained, kappa, checkpoint) in BACKBONES.items():
        common = ['--model', model, '--pretrained', pretrained]
        if checkpoint:
            common += ['--remoteclip', str(models/checkpoint)]
        cache = out/'caches'/tag
        for split in ('train', 'test'):
            path = cache/(split+'.pt')
            add('cache', f'{tag}_{split}', [python, repo/'code/cache_primitives.py', '--ids', clean/(split+'_ids.json'),
                 '--img_root', data/'SARimages_pre', '--primitive_mode', 'physics', '--max_regions', '6', '--max_sc', '8',
                 '--encode_bs', '128', *common, '--out', path], [path])
        for seed in range(5):
            for variant, flags in VARIANTS.items():
                folder = out/'results'/tag/f'seed_{seed}'/variant
                add('train', f'{tag}_{seed}_{variant}', [python, HERE/'training_entry.py', '--repo', repo,
                    '--cache', cache/'train.pt', '--captions', clean/'Caption_train.csv', *common,
                    '--epochs', '10', '--bs', '64', '--text_bs', '256', '--lr', '.002', '--kappa', str(kappa),
                    '--weight_decay', '.0001', '--seed', str(seed), '--ablate', flags, '--out', folder/'head.pt'], [folder/'head.pt'])
                for label, factor in ([('test', '1'), ('spacing_1p5', '1.5'), ('spacing_2', '2')] if tag == 'vitb32' else [('test', '1')]):
                    evalcache = cache/('test.pt' if factor == '1' else label+'.pt')
                    add('evaluate' if factor == '1' else 'evaluate-scales', f'{tag}_{seed}_{variant}_{label}',
                        [python, HERE/'evaluate.py', '--repo', repo, '--cache', evalcache, '--reference-cache', cache/'test.pt',
                         '--captions', clean/'Caption_test.csv', '--head', folder/'head.pt', '--backbone', tag,
                         '--seed', str(seed), '--variant', variant, '--label', label, '--factor', factor, '--device', device,
                         *common, '--out', folder/(label+'.json')], [folder/(label+'.json')])
        if tag == 'vitb32':
            for label, factor in [('spacing_1p5', '1.5'), ('spacing_2', '2')]:
                path = cache/(label+'.pt')
                add('scale-cache', label, [python, repo/'tools/scale/stress_cache.py', '--code-root', repo/'code',
                     '--reference-cache', cache/'test.pt', '--ids', clean/'test_ids.json', '--img-root', data/'SARimages_pre',
                     '--factor', factor, '--expected-gallery', '3688', '--encode-bs', '128',
                     '--device', 'cpu' if device == 'cpu' else 'cuda', *common, '--out', path], [path])
    return jobs


def validate_repo(repo):
    bindings = json.loads((HERE/'SOURCE_REQUIREMENTS.json').read_text())
    for relative, expected in bindings['repository_files'].items():
        if sha(repo/relative) != expected:
            raise ValueError('Required repository code differs: ' + relative)
    return bindings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--hf-home', type=Path, help='The same Hugging Face cache used during asset preparation; default: MODELS/huggingface')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0', choices=('cuda:0', 'cpu'))
    p.add_argument('--stage', required=True, choices=('plan', 'preflight', 'cache', 'train', 'evaluate', 'scale-cache', 'evaluate-scales'))
    a = p.parse_args()
    a.repo, a.data, a.models, a.out = [x.resolve() for x in (a.repo, a.data, a.models, a.out)]
    a.hf_home = (a.hf_home or a.models/'huggingface').resolve()
    bindings = validate_repo(a.repo)
    jobs = build_plan(a.repo, a.data, a.models, a.out, device=a.device)
    if a.stage == 'plan':
        print(json.dumps({'kind': 'new_portable_reproduction_not_archived_measurements', 'hf_home': str(a.hf_home), 'jobs': jobs}, indent=2))
        return
    if a.device != 'cpu' and (not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']):
        p.error('Select exactly one available physical GPU with CUDA_VISIBLE_DEVICES before running')
    # Exact train/test ID checks are independent of any archived cache or anchor.
    for split, count in [('train', 9181), ('test', 3688)]:
        own = a.data/'ret_clean_20261007'/(split+'_ids.json')
        reference = a.repo/'data/splits_content_disjoint_20261007'/(split+'_ids.json')
        ids = json.loads(own.read_text())
        if ids != json.loads(reference.read_text()) or len(ids) != count or len(set(ids)) != count:
            raise ValueError('Wrong retained dataset IDs or order: ' + split)
        caption_file = a.data/'ret_clean_20261007'/('Caption_'+split+'.csv')
        if sha(caption_file) != bindings['clean_caption_sha256']['Caption_'+split+'.csv']:
            raise ValueError('Retained caption bytes differ from the recorded split: ' + split)
        with caption_file.open() as stream:
            reader = csv.DictReader(stream)
            image_key = reader.fieldnames[0]
            caption_key = next(k for k in reader.fieldnames if 'caption' in k.lower() and 'type' not in k.lower())
            valid = set(ids)
            pairs = [(row[image_key].strip(), row[caption_key].strip()) for row in reader
                     if row[image_key].strip() in valid and row[caption_key].strip()]
        if len(pairs) != {'train': 31945, 'test': 6190}[split] or {n for n, _ in pairs} != valid:
            raise ValueError('Wrong caption population: ' + split)
    env = {**os.environ, 'RS_RAG_ROOT': str(a.repo), 'RS_RAG_DATA_ROOT': str(a.data),
           'RS_RAG_MODEL_ROOT': str(a.models), 'PARE_DEVICE': a.device, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'HF_HOME': str(a.hf_home), 'HF_HUB_CACHE': str(a.hf_home/'hub'),
           'HUGGINGFACE_HUB_CACHE': str(a.hf_home/'hub'),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    # Resolve tokenizers under this exact HF_HOME with network disabled before
    # launching any expensive stage. In particular, SigLIP requires its cached
    # tokenizer assets even if the backbone weight is already available.
    tokenizer_check = 'import open_clip; ' + '; '.join(
        f'open_clip.get_tokenizer({model!r})' for model in sorted({b[0] for b in BACKBONES.values()}))
    subprocess.run([sys.executable, '-c', tokenizer_check], cwd=a.repo, env=env, check=True)
    if a.stage == 'preflight':
        print('Dataset identities and offline tokenizer loading passed. No training performed.')
        return
    for job in jobs:
        if job['stage'] != a.stage:
            continue
        if any(Path(s).exists() for s in job['outputs']):
            raise FileExistsError('Refusing to overwrite: ' + job['name'])
        logdir = a.out/'portable_execution_records'
        logdir.mkdir(parents=True, exist_ok=True)
        record = logdir/(job['name']+'.json')
        logfile = logdir/(job['name']+'.log')
        if record.exists() or logfile.exists():
            raise FileExistsError('Prior attempt retained; inspect before retry: ' + job['name'])
        state = {**job, 'status': 'running', 'runner_sha256': sha(__file__),
                 'source_requirements_sha256': sha(HERE/'SOURCE_REQUIREMENTS.json'),
                 'attribution': 'new portable reproduction; not the archived measurement executor'}
        record.write_text(json.dumps(state, indent=2))
        with logfile.open('x') as stream:
            result = subprocess.run(job['argv'], cwd=a.repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        state.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode,
                     artifacts={s: sha(s) for s in job['outputs'] if Path(s).exists()})
        record.write_text(json.dumps(state, indent=2))
        if result.returncode:
            raise RuntimeError('Failed attempt retained: ' + job['name'])
        print(job['name'], 'complete', flush=True)


if __name__ == '__main__':
    main()
