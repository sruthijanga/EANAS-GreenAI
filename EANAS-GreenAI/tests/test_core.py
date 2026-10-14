import random,torch
from eanas.core import arch_a5,random_arch,mutate,crossover,EANASNet,Architecture
from eanas.optimization import Objective,pareto_front,select
from eanas.profiling import integrate_power,CalibratedEnergy,measure
from eanas.stats import summarize,holm

def test_model_a5():
    m=EANASNet(arch_a5());assert m(torch.randn(2,3,32,32)).shape==(2,10)

def test_random_arch():
    rng=random.Random(42);a=random_arch(rng);b=random_arch(rng)
    for x in [a,mutate(a,rng),crossover(a,b,rng)]:
        assert EANASNet(x)(torch.randn(1,3,32,32)).shape==(1,10)
        assert Architecture.from_dict(x.to_dict()).uid==x.uid

def test_selection():
    rs=[dict(uid='a',accuracy=.9,latency_ms=10,energy_j=.1),dict(uid='b',accuracy=.8,latency_ms=20,energy_j=.2),dict(uid='c',accuracy=.95,latency_ms=20,energy_j=.3)]
    assert [r['uid'] for r in pareto_front(rs)]==['a','c']
    assert select(rs,Objective())['uid']=='a'

def test_power():
    assert abs(integrate_power([(0,3),(1,3),(2,3)],1,0,2)-4)<1e-8
    assert CalibratedEnergy(2).attach({'latency_ms':10})['energy_j']==.02

def test_stats():
    x=summarize([1,2,3,4,5]);assert x['n']==5 and x['ci95_low']<x['mean']<x['ci95_high']
    assert len(holm([.01,.02,.5]))==3

def test_profile():
    r=measure(EANASNet(arch_a5()),warmup=1,runs=2)
    assert r['energy_j'] is None and r['latency_ms']>0
