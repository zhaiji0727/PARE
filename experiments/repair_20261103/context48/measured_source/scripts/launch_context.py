#!/usr/bin/env python3
"""Explicit three-seed launcher; no automatic progression to internal checks."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import run_context as run
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','check'],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--retry-incomplete',action='store_true');args=p.parse_args()
    run.verify_freeze(argparse.Namespace(freeze=ROOT/'CONTEXT_SOURCE_FROZEN.json',split_dir=ROOT/'plans/splits'))
    if not (ROOT/'INPUTS_FROZEN.json').exists():raise ValueError('Bind the verified complete overlay before launching')
    if args.stage=='check' and not (ROOT/'results/TRAINING_COMPLETE_LOCK.json').exists():raise ValueError('All nine fits and mandatory A0 anchors must complete first')
    commands=[]
    for seed in [0,1,2]:
        cmd=[sys.executable,str(ROOT/'scripts/run_context.py'),args.stage,'--seed',str(seed),'--gpu',str(seed)]
        if args.resume:cmd.append('--resume')
        if args.retry_incomplete:cmd.append('--retry-incomplete')
        commands.append({'seed':seed,'gpu':seed,'argv':cmd})
    if not args.execute:print(json.dumps(commands,indent=2));return
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    states={int(a[0]):(int(a[1]),int(a[2])) for line in raw.splitlines() if (a:=line.split(','))}
    if any(states[s][0]>256 or states[s][1]>0 for s in [0,1,2]):raise ValueError('One of GPUs0–2 is occupied')
    folder=ROOT/'logs';folder.mkdir(exist_ok=True);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');record={'stage':args.stage,'started_utc':stamp,'processes':[]}
    for row in commands:
        log=folder/f'{args.stage}_seed_{row["seed"]}_{stamp}.log'
        with log.open('x') as stream:proc=subprocess.Popen(row['argv'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record['processes'].append({**row,'pid':proc.pid,'log':str(log)})
        (folder/f'launch_{args.stage}_{stamp}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
