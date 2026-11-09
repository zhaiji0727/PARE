#!/usr/bin/env python3
"""Apply recorded numerical environment, then run original repository trainer."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--repo', type=Path, required=True)
    a, remaining = p.parse_known_args()
    code = a.repo.resolve()/'code'
    sys.path.insert(0, str(code))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # No algorithm, hyperparameter, optimizer, loss, RNG, or pair-order patch.
    sys.argv = [str(code/'train_saqre.py'), *remaining]
    runpy.run_path(str(code/'train_saqre.py'), run_name='__main__')


if __name__ == '__main__':
    main()
