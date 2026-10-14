#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eanas.utils import load_yaml
from eanas.data import loaders
from eanas.search import SearchConfig,HybridSearcher
from eanas.optimization import Objective
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/search.yaml');p.add_argument('--smoke',action='store_true');p.add_argument('--synthetic',action='store_true')
p.add_argument('--energy-mode',choices=['measured','calibrated']);p.add_argument('--calibrated-dynamic-power-w',type=float);p.add_argument('--power-csv');p.add_argument('--device')
args=p.parse_args();conf=load_yaml(args.config);s=conf['search'].copy();o=conf['objective'].copy();data=conf.get('data',{})
for key,val in [('energy_mode',args.energy_mode),('calibrated_dynamic_power_w',args.calibrated_dynamic_power_w),('power_csv',args.power_csv),('device',args.device)]:
    if val is not None:s[key]=val
if args.smoke:s.update(iterations=2,population_size=3,max_evaluations=6,search_epochs=1,profile_warmup=1,profile_runs=3,max_train_batches=1,max_val_batches=1,output_dir='results/smoke')
train,val,_=loaders(root=data.get('root','data'),batch_size=s['batch_size'],workers=data.get('workers',0),seed=s.get('seed',42),synthetic=args.synthetic or data.get('synthetic',False),train_samples=96 if args.smoke else 256,val_samples=32 if args.smoke else 64)
selected=HybridSearcher(SearchConfig(**s),Objective(**o),train,val).run();print(json.dumps(selected,indent=2))
