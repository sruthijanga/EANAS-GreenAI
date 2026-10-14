#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eanas.core import Architecture,arch_a5,EANASNet
from eanas.baselines import get_baseline
from eanas.data import loaders
from eanas.training import fit,accuracy,seed_all
from eanas.utils import write_json
p=argparse.ArgumentParser();p.add_argument('--architecture',default='arch_a5');p.add_argument('--dataset',choices=['cifar10','imagenet'],default='cifar10');p.add_argument('--data-root',default='data');p.add_argument('--epochs',type=int,default=None);p.add_argument('--batch-size',type=int,default=None);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=2);p.add_argument('--synthetic',action='store_true');p.add_argument('--max-batches',type=int);p.add_argument('--out',default='results/final')
a=p.parse_args();seed_all(a.seed);batch=a.batch_size or (256 if a.dataset=='imagenet' else 128)
train,val,test=loaders(a.dataset,a.data_root,batch,a.workers,a.seed,synthetic=a.synthetic)
if a.architecture=='arch_a5': model=EANASNet(arch_a5(),1000 if a.dataset=='imagenet' else 10)
elif Path(a.architecture).exists():
    spec=json.loads(Path(a.architecture).read_text());model=EANASNet(Architecture.from_dict(spec.get('arch',spec)),1000 if a.dataset=='imagenet' else 10)
else:model=get_baseline(a.architecture,1000 if a.dataset=='imagenet' else 10)
out=Path(a.out)/f'{a.dataset}_{a.architecture.replace("/","_")}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
r=fit(model,train,val,epochs=a.epochs or (100 if a.dataset=='imagenet' else 200),lr=.1 if a.dataset=='cifar10' else .1,device=a.device,max_train_batches=a.max_batches,max_val_batches=a.max_batches,checkpoint=str(out/'best.pt'))
r['test_accuracy']=accuracy(model,test,a.device,a.max_batches);r['seed']=a.seed;r['dataset']=a.dataset
write_json(out/'metrics.json',r);print(json.dumps({'val_accuracy':r['val_accuracy'],'test_accuracy':r['test_accuracy'],'output':str(out)},indent=2))
