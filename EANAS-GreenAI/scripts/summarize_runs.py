#!/usr/bin/env python3
"""Aggregate genuinely independent run-level CSV observations."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eanas.stats import summarize,compare_paired,holm
p=argparse.ArgumentParser();p.add_argument('--csv',required=True,help='CSV with columns model,seed,accuracy,latency_ms,energy_j');p.add_argument('--reference',default='EANAS');p.add_argument('--out',default='results/statistics.json');a=p.parse_args()
with open(a.csv,newline='') as f:rows=list(csv.DictReader(f))
models={m for m in (r['model'] for r in rows)};metrics=('accuracy','latency_ms','energy_j');res={}
for model in sorted(models):
    group=[r for r in rows if r['model']==model];res[model]={metric:summarize([float(r[metric]) for r in group]) for metric in metrics}
if a.reference in models:
    res['paired_comparisons']={}
    for metric in metrics:
        ref={r['seed']:float(r[metric]) for r in rows if r['model']==a.reference}
        comparisons=[]
        for other in sorted(models-{a.reference}):
            alt={r['seed']:float(r[metric]) for r in rows if r['model']==other};common=sorted(set(ref)&set(alt))
            if len(common)>1:
                item=compare_paired([ref[k] for k in common],[alt[k] for k in common]);item['model']=other;item['n_pairs']=len(common);comparisons.append(item)
        adjusted=holm([x['p_value'] for x in comparisons]);
        for item,padj in zip(comparisons,adjusted):item['holm_p_value']=padj
        res['paired_comparisons'][metric]=comparisons
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(res,indent=2));print(a.out)
