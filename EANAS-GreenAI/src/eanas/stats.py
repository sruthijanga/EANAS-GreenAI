import math
import numpy as np
from scipy import stats

def summarize(values):
    x=np.asarray(values,dtype=float)
    if len(x)<2:raise ValueError('At least two independent observations required')
    mean=float(x.mean());sd=float(x.std(ddof=1));margin=float(stats.t.ppf(.975,len(x)-1)*sd/math.sqrt(len(x)))
    return dict(n=len(x),mean=mean,sd=sd,ci95_low=mean-margin,ci95_high=mean+margin)

def compare_paired(reference,alternative):
    a=np.asarray(reference,float);b=np.asarray(alternative,float)
    if a.shape!=b.shape or len(a)<2:raise ValueError('Must provide matched independent observations')
    difference=a-b
    if np.isclose(difference.std(ddof=1),0):
        return dict(mean_difference=float(difference.mean()),p_value=0.0 if difference.mean()!=0 else 1.0,effect_size=None)
    test=stats.ttest_rel(a,b)
    return dict(mean_difference=float(difference.mean()),p_value=float(test.pvalue),effect_size=float(difference.mean()/difference.std(ddof=1)))

def holm(pvalues):
    p=np.asarray(pvalues,float);n=len(p); order=np.argsort(p);adjusted=np.empty(n);prev=0.
    for i,idx in enumerate(order):
        prev=max(prev,min(1.,(n-i)*p[idx]));adjusted[idx]=prev
    return adjusted.tolist()
