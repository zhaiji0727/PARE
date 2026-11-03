"""Tiny synthetic optimizer/ordering check against the frozen loop; no data files."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-code', required=True, type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    sys.path[:0] = [str(args.base_code.resolve()), str(Path(__file__).resolve().parents[1] / 'code')]
    from saqre_phys import SAQREHead
    from factorial_head import FactorialHead, Factors
    from train_saqre import move_docs
    from batch_scoring import score_matrix
    from run_factorial_fitval import train_one, object_sha
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = args.device
    settings = dict(lr=.002, kappa=.03, weight_decay=.0001, epochs=2,
                    batch_size=4, batches_per_epoch=3, tail_batch_pairs=1)
    torch.manual_seed(77)
    dim, count = 32, 5
    cache = {'scene': F.normalize(torch.randn(count, dim), dim=-1),
             'agg': torch.randn(count, dim), 'nu': torch.rand(count), 'Mc': 3, 'Mf': 4}
    for prefix, regions in [('c', 3), ('f', 4)]:
        cache[prefix + '_emb'] = F.normalize(torch.randn(count, regions, dim), dim=-1)
        cache[prefix + '_mu'] = torch.rand(count, regions)
        cache[prefix + '_t'] = torch.rand(count, regions)
        cache[prefix + '_tf'] = torch.rand(count, regions, 4)
        cache[prefix + '_n'] = torch.arange(count) % (regions + 1)
    # Repeated image rows and caption identities exercise the original diagonal
    # loss's duplicate handling, including the final size-one batch.
    pairs = [(0, 'a'), (0, 'b'), (1, 'c'), (2, 'd'), (2, 'a'), (3, 'e'), (4, 'f'), (1, 'g'), (3, 'h')]
    vectors = {letter: F.normalize(torch.randn(dim), dim=-1) for letter in 'abcdefgh'}
    results = []
    for seed, (abc, flag), (r, t) in itertools.product(
            [0, 1, 2], [((1, 1, 1), None), ((0, 0, 0), 'cfar_strict'), ((0, 1, 1), 'cfar')],
            list(itertools.product([0, 1], repeat=2))):
        torch.manual_seed(seed)
        old = SAQREHead(dim).to(device)
        old.ablate = ({flag} if flag else set()) | ({'sr'} if not r else set()) | ({'ty'} if not t else set())
        torch.manual_seed(seed)
        new = FactorialHead(dim, Factors(*abc, r, t)).to(device)
        rng, order, expected = random.Random(seed), list(range(len(pairs))), []
        for _ in range(settings['epochs']):
            rng.shuffle(order)
            expected.append(object_sha(order))
        history = train_one(new, cache, pairs, vectors, device, seed, settings, expected, lambda _: None)
        opt = torch.optim.AdamW(old.parameters(), lr=.002, weight_decay=.0001)
        plist, rng, reference_losses = pairs[:], random.Random(seed), []
        for _ in range(settings['epochs']):
            rng.shuffle(plist)
            total = 0.
            for start in range(0, len(plist), 4):
                batch = plist[start:start + 4]
                rows = torch.tensor([row for row, _ in batch])
                q = torch.stack([vectors[c] for _, c in batch]).to(device)
                scores = score_matrix(old, q, move_docs(cache, rows, device))
                loss = F.cross_entropy(scores / .03, torch.arange(len(batch), device=device))
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            reference_losses.append(total / 3)
        max_error = 0.
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], atol=1e-7, rtol=0, msg=key)
            max_error = max(max_error, float((value - new.state_dict()[key]).abs().max()))
        assert reference_losses == [row['mean_batch_loss'] for row in history]
        results.append({'seed': seed, 'factors': new.factors.to_dict(), 'max_parameter_error': max_error,
                        'losses_exact': True, 'pair_order_sha256': expected})
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'all_passed': True, 'device': device, 'cases': results}, indent=2) + '\n')
    print(json.dumps({'all_passed': True, 'cases': len(results), 'max_parameter_error': max(x['max_parameter_error'] for x in results)}))


if __name__ == '__main__':
    main()
