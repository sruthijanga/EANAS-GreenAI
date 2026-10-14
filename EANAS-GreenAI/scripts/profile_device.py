#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eanas.core import EANASNet,arch_a5
from eanas.profiling import measure,CalibratedEnergy
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--device',default='cpu');p.add_argument('--power-csv');p.add_argument('--idle-w',type=float,default=0.);p.add_argument('--calibrated-power-w',type=float);p.add_argument('--warmup',type=int,default=50);p.add_argument('--runs',type=int,default=1000);p.add_argument('--out',default='results/profiling.json');a=p.parse_args()
import torch
m=EANASNet(arch_a5());
if a.checkpoint:
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(ckpt.get('model',ckpt))
r=measure(m,device=a.device,warmup=a.warmup,runs=a.runs,power_csv=a.power_csv,idle_w=a.idle_w)
if a.calibrated_power_w is not None:
    if a.power_csv:raise ValueError('Choose real measured power or calibration, not both')
    r=CalibratedEnergy(a.calibrated_power_w).attach(r)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
