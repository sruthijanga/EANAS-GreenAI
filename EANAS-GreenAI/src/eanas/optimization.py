"""Constrained normalized objective and exact Pareto sorting."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Objective:
    accuracy_weight:float=.5
    latency_weight:float=.25
    energy_weight:float=.25
    latency_ref_ms:float=25.0
    energy_ref_j:float=1.0
    max_latency_ms:float|None=None
    max_energy_j:float|None=None
    max_memory_mb:float|None=None
    def __post_init__(self):
        if abs(self.accuracy_weight+self.latency_weight+self.energy_weight-1)>1e-8: raise ValueError('Objective weights must sum to 1')
        if self.latency_ref_ms<=0 or self.energy_ref_j<=0:raise ValueError('Reference values must be positive')
    def feasible(self,r):
        for key,lim in [('latency_ms',self.max_latency_ms),('energy_j',self.max_energy_j),('memory_mb',self.max_memory_mb)]:
            if lim is not None and (r.get(key) is None or r[key]>lim):return False
        return True
    def score(self,r):
        if r.get('energy_j') is None: raise ValueError('Measured or explicitly estimated energy is necessary to score energy-aware search')
        return (self.accuracy_weight*(1-r['accuracy'])+self.latency_weight*r['latency_ms']/self.latency_ref_ms+self.energy_weight*r['energy_j']/self.energy_ref_j)

def dominates(a,b):
    av=(1-a['accuracy'],a['latency_ms'],a['energy_j']); bv=(1-b['accuracy'],b['latency_ms'],b['energy_j'])
    return all(x<=y for x,y in zip(av,bv)) and any(x<y for x,y in zip(av,bv))

def pareto_front(results,obj=None):
    eligible=[r for r in results if r.get('energy_j') is not None and (obj is None or obj.feasible(r))]
    return [r for r in eligible if not any(dominates(q,r) for q in eligible if q is not r)]

def select(results,obj):
    front=pareto_front(results,obj)
    if not front:raise ValueError('No feasible, fully measured candidates')
    return min(front,key=lambda r:(obj.score(r),r['uid']))
