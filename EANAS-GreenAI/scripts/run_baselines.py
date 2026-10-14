#!/usr/bin/env python3
import subprocess,sys,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['mobilenetv2','efficientnet_b0']);p.add_argument('--epochs',type=int,default=200);p.add_argument('--synthetic',action='store_true');p.add_argument('--max-batches',type=int);a=p.parse_args()
for model in a.models:
    cmd=[sys.executable,str(Path(__file__).with_name('train_final.py')),'--architecture',model,'--epochs',str(a.epochs)]
    if a.synthetic:cmd.append('--synthetic')
    if a.max_batches:cmd+=['--max-batches',str(a.max_batches)]
    subprocess.run(cmd,check=True)
