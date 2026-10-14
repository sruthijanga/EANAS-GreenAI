"""REINFORCE controller, evolutionary search and real candidate training."""
import csv,json,random,math
from pathlib import Path
from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Categorical
from .core import OPS,WIDTHS,Architecture,BlockSpec,EANASNet,random_arch,mutate,crossover
from .training import fit,seed_all
from .profiling import measure,CalibratedEnergy
from .optimization import Objective,select,pareto_front

class LSTMController(nn.Module):
    def __init__(self,hidden=128,layers=2):
        super().__init__();self.emb=nn.Embedding(len(OPS)+1,hidden);self.lstm=nn.LSTM(hidden,hidden,layers,batch_first=True)
        self.op_head=nn.Linear(hidden,len(OPS));self.width_head=nn.Linear(hidden,len(WIDTHS));self.stride_head=nn.Linear(hidden,2)
    def sample(self,device='cpu'):
        tokens=torch.tensor([[len(OPS)]],device=device); state=None; decisions=[]; logps=[];entropies=[];ch=32
        for i in range(5):
            out,state=self.lstm(self.emb(tokens),state);v=out[:,-1,:]
            op_d=Categorical(logits=self.op_head(v));op_idx=op_d.sample();logps.append(op_d.log_prob(op_idx));entropies.append(op_d.entropy())
            if i in (1,3):
                wd=Categorical(logits=self.width_head(v));idx=wd.sample();ch=WIDTHS[idx.item()];logps.append(wd.log_prob(idx));entropies.append(wd.entropy())
                sd=Categorical(logits=self.stride_head(v));sidx=sd.sample();stride=sidx.item()+1;logps.append(sd.log_prob(sidx));entropies.append(sd.entropy())
            else:stride=1
            decisions.append(BlockSpec(OPS[op_idx.item()],ch,stride));tokens=op_idx.reshape(1,1)
        return Architecture(tuple(decisions)),torch.stack(logps).sum(),torch.stack(entropies).sum()

@dataclass
class SearchConfig:
    iterations:int=50
    population_size:int=20
    max_evaluations:int=1000
    search_epochs:int=10
    mutation_prob:float=.2
    crossover_prob:float=.8
    tournament_size:int=3
    elite_count:int=2
    batch_size:int=128
    controller_lr:float=.00035
    controller_hidden:int=128
    device:str='cpu'
    profile_warmup:int=50
    profile_runs:int=1000
    max_train_batches:int|None=None
    max_val_batches:int|None=None
    energy_mode:str='measured' # measured, calibrated; measured requires attached power CSV
    calibrated_dynamic_power_w:float|None=None
    power_csv:str|None=None
    output_dir:str='results/search'
    seed:int=42

class HybridSearcher:
    def __init__(self,cfg,objective:Objective,train,val):
        self.cfg=cfg;self.obj=objective;self.train=train;self.val=val
        self.rng=random.Random(cfg.seed);seed_all(cfg.seed)
        self.policy=LSTMController(cfg.controller_hidden).to(cfg.device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=cfg.controller_lr)
        self.ema=None;self.cache={};self.history=[];self.population=[]
        self.enable_rl=True;self.enable_ea=True;self.use_hardware_feedback=True;self.use_multiobjective=True
        Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
        if cfg.energy_mode=='calibrated' and cfg.calibrated_dynamic_power_w is None:raise ValueError('calibrated_dynamic_power_w required')
    def eval_arch(self,arch):
        if arch.uid in self.cache:return self.cache[arch.uid]
        if len(self.cache)>=self.cfg.max_evaluations:return None
        model=EANASNet(arch)
        trained=fit(model,self.train,self.val,epochs=self.cfg.search_epochs,device=self.cfg.device,
                    max_train_batches=self.cfg.max_train_batches,max_val_batches=self.cfg.max_val_batches)
        profile=measure(model,device=self.cfg.device,warmup=self.cfg.profile_warmup,runs=self.cfg.profile_runs,power_csv=self.cfg.power_csv)
        if self.cfg.energy_mode=='calibrated':profile=CalibratedEnergy(self.cfg.calibrated_dynamic_power_w).attach(profile)
        r={'uid':arch.uid,'arch':arch.to_dict(),'accuracy':trained['val_accuracy'],**profile,'generation':len(self.history)}
        r['feasible']=self.obj.feasible(r)
        r['score']=self.search_score(r) if r['energy_j'] is not None else None
        self.cache[arch.uid]=r
        with open(Path(self.cfg.output_dir)/'candidates.jsonl','a') as f:f.write(json.dumps(r)+'\n')
        return r
    def search_score(self,r):
        if not self.use_hardware_feedback or not self.use_multiobjective:
            return 1-r['accuracy']
        return self.obj.score(r)
    def tournament(self):
        subset=self.rng.sample(self.population,min(self.cfg.tournament_size,len(self.population)))
        return min(subset,key=lambda a:self._fitness(a))
    def _fitness(self,arch):
        r=self.cache[arch.uid]
        return r['score'] if r['feasible'] and r['score'] is not None else float('inf')
    def run(self):
        cfg=self.cfg
        if cfg.energy_mode=='measured' and not cfg.power_csv:
            raise RuntimeError('Energy-aware search requires timestamped power CSV from target device or explicitly configured calibrated mode. For initial tests use --energy-mode calibrated --calibrated-dynamic-power-w VALUE; results will be labeled estimated.')
        for generation in range(cfg.iterations):
            if len(self.cache)>=cfg.max_evaluations:break
            candidates=[];policy_items=[]
            
            if self.enable_rl:
                arch,logprob,entropy=self.policy.sample(cfg.device);policy_items.append((arch,logprob,entropy))
            else: arch=random_arch(self.rng)
            candidates.append(arch)
            if not self.enable_ea:
                for _ in range(max(0,cfg.population_size-1)):
                    if self.enable_rl:
                        ac,lp,en=self.policy.sample(cfg.device);policy_items.append((ac,lp,en))
                    else: ac=random_arch(self.rng)
                    candidates.append(ac)
            elif len(self.population)>1:
                for _ in range(max(1,cfg.population_size-1)):
                    a=self.tournament()
                    if self.rng.random()<cfg.crossover_prob:
                        b=self.tournament();child=crossover(a,b,self.rng)
                    else:child=a
                    candidates.append(mutate(child,self.rng,cfg.mutation_prob))
            else:
                candidates.extend(random_arch(self.rng) for _ in range(cfg.population_size-1))
            evaluated=[]
            for c in candidates:
                r=self.eval_arch(c)
                if r is not None:evaluated.append(c)
            losses=[]
            for arch,lp,ent in policy_items:
                r=self.cache.get(arch.uid)
                if self.enable_rl and r and r['score'] is not None:
                    reward= -r['score'] if r['feasible'] else -10-r['score']
                    baseline=self.ema if self.ema is not None else reward
                    losses.append(-(reward-baseline)*lp - .001*ent)
                    self.ema=reward if self.ema is None else .9*self.ema+.1*reward
            if losses:
                loss=torch.stack(losses).mean()
                self.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.policy.parameters(),5.);self.optimizer.step()
            combined={a.uid:a for a in self.population+evaluated}
            sorted_arch=sorted(combined.values(),key=self._fitness)
            # Preserve elites then retain scored population with randomized ties/diversity.
            self.population=sorted_arch[:cfg.population_size]
            finite=[r for r in self.cache.values() if r['score'] is not None and r['feasible']]
            best=min((r['score'] for r in finite),default=None)
            rec={'iteration':generation+1,'evaluated':len(self.cache),'best_score':best}
            self.history.append(rec)
            (Path(cfg.output_dir)/'history.json').write_text(json.dumps(self.history,indent=2))
            torch.save({'controller':self.policy.state_dict(),'optimizer':self.optimizer.state_dict(),'generation':generation+1},Path(cfg.output_dir)/'controller.pt')
        rows=list(self.cache.values());(Path(cfg.output_dir)/'all_results.json').write_text(json.dumps(rows,indent=2))
        front=pareto_front(rows,self.obj)
        if self.use_hardware_feedback and self.use_multiobjective:
            selected=select(rows,self.obj)
        else:
            feasible=[r for r in rows if r['feasible']]
            if not feasible: raise ValueError('No feasible candidates')
            selected=max(feasible,key=lambda r:(r['accuracy'],-r['latency_ms']))
        (Path(cfg.output_dir)/'selected_architecture.json').write_text(json.dumps(selected,indent=2))
        (Path(cfg.output_dir)/'pareto_front.json').write_text(json.dumps(front,indent=2))
        return selected
