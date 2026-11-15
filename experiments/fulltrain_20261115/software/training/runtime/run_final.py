"""Final-study runner; shipped protocol remains unbound and cannot execute."""
import argparse
from pathlib import Path
from protocol_gate import load_protocol


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['train','evaluate'])
    ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--backbone');ap.add_argument('--model');ap.add_argument('--seed',type=int);ap.add_argument('--gpu')
    args=ap.parse_args(argv)
    protocol=load_protocol(args.protocol,args.protocol_sha256,args.stage)
    if args.stage=='train':
        # Lazy import occurs only after the UNBOUND/status/complete-protocol gate.
        from train_backend import execute_training
        return execute_training(protocol,args)
    raise RuntimeError('Final evaluation backend/ranking contract remains unimplemented and unbound; no evaluation performed')

if __name__=='__main__':main()
