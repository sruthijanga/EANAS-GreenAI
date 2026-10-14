#!/usr/bin/env python3
"""Run ablations by applying separate, explicit search-engine configurations."""
import argparse,sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eanas.search import SearchConfig,HybridSearcher
from eanas.optimization import Objective
from eanas.data import loaders
from eanas.utils import load_yaml
from eanas.ablation import AblationSearcher
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/search.yaml');p.add_argument('--variant',choices=['full','without_rl','without_ea','without_hardware_feedback','without_multiobjective'],required=True);p.add_argument('--synthetic',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--energy-mode',choices=['measured','calibrated']);p.add_argument('--calibrated-dynamic-power-w',type=float);p.add_argument('--power-csv');a=p.parse_args()
c=load_yaml(a.config);s=c['search'].copy();s['seed']=a.seed;s['output_dir']=f'results/ablations/{a.variant}/seed_{a.seed}'
if a.energy_mode:s['energy_mode']=a.energy_mode
if a.calibrated_dynamic_power_w is not None:s['calibrated_dynamic_power_w']=a.calibrated_dynamic_power_w
if a.power_csv:s['power_csv']=a.power_csv
if a.smoke:s.update(iterations=2,population_size=3,max_evaluations=6,search_epochs=1,profile_warmup=1,profile_runs=3,max_train_batches=1,max_val_batches=1)
tr,va,_=loaders(root=c.get('data',{}).get('root','data'),batch_size=s['batch_size'],workers=0,seed=a.seed,synthetic=a.synthetic or a.smoke)
obj=Objective(**c['objective']);search=AblationSearcher(SearchConfig(**s),obj,tr,va,variant=a.variant)
print(json.dumps(search.run(),indent=2))
