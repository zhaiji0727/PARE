"""Small synthetic score/gradient checks; never reads retrieval results."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-code', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path[:0] = [str(args.base_code.resolve()), str(Path(__file__).resolve().parents[1] / 'code')]
    from saqre_phys import SAQREHead
    from factorial_head import FactorialHead, Factors
    from batch_scoring import score_matrix
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)

    def docs(dim, mask_mode):
        n = 7
        result = {'g': F.normalize(torch.randn(n, dim, device=device), dim=-1),
                  'agg': torch.randn(n, dim, device=device) * .1,
                  'nu': torch.rand(n, device=device) * .1}
        for prefix, count in [('c', 6), ('f', 8)]:
            result[prefix + '_emb'] = F.normalize(torch.randn(n, count, dim, device=device), dim=-1)
            result[prefix + '_mu'] = torch.rand(n, count, device=device)
            result[prefix + '_t'] = torch.rand(n, count, device=device)
            result[prefix + '_tf'] = torch.randn(n, count, 4, device=device)
            counts = torch.arange(n, device=device) % (count + 1) if mask_mode == 'mixed' else torch.full((n,), count, device=device)
            if mask_mode == 'empty_' + prefix or mask_mode == 'empty_both':
                counts.zero_()
            result[prefix + '_mask'] = torch.arange(count, device=device)[None, :] < counts[:, None]
        return result

    rows = []
    for dim, mask_mode in itertools.product([512, 768], ['full', 'mixed', 'empty_c', 'empty_f', 'empty_both']):
        torch.manual_seed(17)
        document = docs(dim, mask_mode)
        query = F.normalize(torch.randn(5, dim, device=device), dim=-1)
        for abc, original_flag in [((1, 1, 1), None), ((0, 0, 0), 'cfar_strict'), ((0, 1, 1), 'cfar')]:
            for reliability, token in itertools.product([0, 1], repeat=2):
                torch.manual_seed(31)
                old = SAQREHead(dim).to(device)
                old.ablate = ({original_flag} if original_flag else set()) | ({'sr'} if not reliability else set()) | ({'ty'} if not token else set())
                torch.manual_seed(31)
                new = FactorialHead(dim, Factors(*abc, reliability, token)).to(device)
                assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
                expected = score_matrix(old, query, document)
                actual = score_matrix(new, query, document)
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
                # Fixed arbitrary score weights avoid masking gradient differences
                # through the symmetry of a summed loss. Also compare CE gradients.
                weights = torch.arange(actual.numel(), device=device).reshape_as(actual) / actual.numel()
                for name, loss_old, loss_new in [
                    ('weighted_score', (expected * weights).sum(), (actual * weights).sum()),
                    ('contrastive', F.cross_entropy(expected / .03, torch.arange(5, device=device)),
                     F.cross_entropy(actual / .03, torch.arange(5, device=device)))]:
                    old.zero_grad(set_to_none=True)
                    new.zero_grad(set_to_none=True)
                    loss_old.backward(retain_graph=True)
                    loss_new.backward(retain_graph=True)
                    maximum = 0.
                    for (key, left), (right_key, right) in zip(old.named_parameters(), new.named_parameters()):
                        assert key == right_key
                        assert (left.grad is None) == (right.grad is None), key
                        if left.grad is not None:
                            assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()
                            torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=0, msg=key)
                            maximum = max(maximum, float((left.grad - right.grad).abs().max()))
                    rows.append({'D': dim, 'masks': mask_mode, 'factors': new.factors.to_dict(), 'reference_flags': sorted(old.ablate),
                                 'loss': name, 'max_score_error': float((actual - expected).abs().max()), 'max_parameter_gradient_error': maximum})
    result = {'all_passed': True, 'device': str(device), 'tolerance_absolute': 1e-7,
              'cases': rows, 'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
                  'base_saqre_phys.py': args.base_code / 'saqre_phys.py',
                  'batch_scoring.py': args.base_code / 'batch_scoring.py',
                  'factorial_head.py': Path(__file__).resolve().parents[1] / 'code/factorial_head.py',
                  'checker': Path(__file__)}.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': True, 'comparisons': len(rows), 'max_score_error': max(row['max_score_error'] for row in rows),
                      'max_parameter_gradient_error': max(row['max_parameter_gradient_error'] for row in rows)}))


if __name__ == '__main__':
    main()
