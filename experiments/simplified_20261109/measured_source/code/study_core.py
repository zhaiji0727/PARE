"""Identity, pairing and ranking contract for the prespecified 75-head study."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

SEEDS = [0, 1, 2, 3, 4]
VARIANTS = {'full': [], 'strict': ['cfar_strict'], 'strict_token_off': ['cfar_strict', 'ty']}
BACKBONES = ['vitb32', 'vitb16', 'remoteclip', 'siglip', 'georsclip']
SETTINGS = dict(epochs=10, batch_size=64, weight_decay=.0001, n_train_images=9181,
                n_train_pairs=31945, batches_per_epoch=500, tail_batch_pairs=9,
                optimizer='AdamW', precision='float32')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def textsha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

def same_tensor(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.detach().cpu().contiguous().numpy().tobytes() == b.detach().cpu().contiguous().numpy().tobytes()

def validate_plan(plan):
    if plan['schema'] != 'pare.simplified-matched-study.v1' or plan['seeds'] != SEEDS or plan['variants'] != VARIANTS:
        raise ValueError('The fixed five-seed/three-variant protocol differs')
    if set(plan['backbones']) != set(BACKBONES) or plan['expected_heads'] != 75:
        raise ValueError('Exactly five backbones and 75 heads required')
    if any(plan['training'].get(k) != v for k,v in SETTINGS.items()):
        raise ValueError('The retained-train recipe differs')
    if any(plan['training'].get(k, False) for k in ('amp','early_stopping','warm_start','new_hyperparameter_search')):
        raise ValueError('No selection or alternate training is allowed')
    expected = {'vitb32':('ViT-B-32','laion2b_s34b_b79k',.03), 'vitb16':('ViT-B-16','laion2b_s34b_b88k',.05),
                'remoteclip':('ViT-B-32','laion2b_s34b_b79k',.05), 'siglip':('ViT-B-16-SigLIP','webli',.07),
                'georsclip':('ViT-B-32','openai',.05)}
    for tag, (model, pretrained, kappa) in expected.items():
        b = plan['backbones'][tag]
        if (b['model'],b['pretrained'],b['lr'],b['kappa']) != (model,pretrained,.002,kappa):
            raise ValueError('Previously selected backbone/hyperparameters changed')
    e = plan['evaluation']
    if (e['n_queries'],e['n_gallery'],e['main_evaluations'],e['scale_backbone'],e['scale_factors'],e['scale_seeds'],e['scale_variants'],e['additional_scale_evaluations']) != (3688,3688,75,'vitb32',[1.5,2.0],SEEDS,list(VARIANTS),30):
        raise ValueError('The fixed evaluation grid differs')

def training_pairs(cache_ids, captions):
    index = {name: row for row,name in enumerate(cache_ids)}
    return [(index[name],caption) for name,strings in captions.items() if name in index for caption in strings]

def training_orders(seed, n, epochs):
    order = list(range(n)); rng = random.Random(seed); hashes = []
    for _ in range(epochs):
        rng.shuffle(order); hashes.append(objsha(order))
    return hashes

def rank_metrics(ranks):
    if not ranks or any(type(rank) is not int or rank < 1 or rank > len(ranks) for rank in ranks):
        raise ValueError('Invalid square full-gallery ranks')
    return {**{f'R@{k}':100 * sum(r <= k for r in ranks) / len(ranks) for k in (1,5,10)},
            'MRR@10':100 * sum(1 / r if r <= 10 else 0 for r in ranks) / len(ranks)}

def head_ranks(scores, scene_scores, gold):
    """Exact 07 K=N tie convention, including one-dimensional inner argsort."""
    import torch
    from evaluate_multik import rank_of_gold
    order = torch.argsort(scene_scores, dim=1, descending=True)
    return [rank_of_gold(scores[i,order[i]], order[i].tolist(), target) for i,target in enumerate(gold)]

def evaluation_labels(tag):
    return ['test','spacing_1p5','spacing_2'] if tag == 'vitb32' else ['test']
