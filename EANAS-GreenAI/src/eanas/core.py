"""Searchable EANAS models. Shapes and skip compatibility are checked at construction."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json, random
import torch
from torch import nn

OPS = ('conv3','dwconv3','attention','avgpool','maxpool','skip')
WIDTHS = (32,64,128)

@dataclass(frozen=True)
class BlockSpec:
    op: str
    channels: int
    stride: int = 1
    residual: bool = False
    def __post_init__(self):
        if self.op not in OPS: raise ValueError(f'Unknown operator {self.op}')
        if self.channels not in WIDTHS: raise ValueError('Invalid channel width')
        if self.stride not in (1,2): raise ValueError('Invalid stride')

@dataclass(frozen=True)
class Architecture:
    blocks: tuple[BlockSpec,...]
    stem_channels: int = 32
    def __post_init__(self):
        if len(self.blocks)!=5: raise ValueError('Exactly 5 searchable blocks required')
    def to_dict(self): return {'stem_channels':self.stem_channels,'blocks':[asdict(b) for b in self.blocks]}
    @classmethod
    def from_dict(cls,d): return cls(tuple(BlockSpec(**b) for b in d['blocks']),d.get('stem_channels',32))
    @property
    def uid(self): return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()[:16]

class ConvBNAct(nn.Sequential):
    def __init__(self,ci,co,k=3,stride=1,groups=1):
        super().__init__(nn.Conv2d(ci,co,k,stride,padding=k//2,groups=groups,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True))

class DWConv(nn.Sequential):
    def __init__(self,ci,co,stride=1):
        super().__init__(ConvBNAct(ci,ci,3,stride,ci),ConvBNAct(ci,co,1))

class SEAttention(nn.Module):
    def __init__(self,ci,co,stride=1):
        super().__init__(); self.project=ConvBNAct(ci,co,1,stride) if ci!=co or stride!=1 else nn.Identity()
        h=max(8,co//8); self.gate=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(co,h,1),nn.ReLU(True),nn.Conv2d(h,co,1),nn.Sigmoid())
    def forward(self,x):
        x=self.project(x); return x*self.gate(x)

class SearchBlock(nn.Module):
    def __init__(self,ci,spec):
        super().__init__(); co=spec.channels; s=spec.stride
        if spec.op=='conv3': self.op=ConvBNAct(ci,co,3,s)
        elif spec.op=='dwconv3': self.op=DWConv(ci,co,s)
        elif spec.op=='attention': self.op=SEAttention(ci,co,s)
        elif spec.op in ('avgpool','maxpool'):
            pool=nn.AvgPool2d(3,s,1) if spec.op=='avgpool' else nn.MaxPool2d(3,s,1)
            self.op=nn.Sequential(pool,ConvBNAct(ci,co,1) if ci!=co else nn.Identity())
        else: self.op=nn.Identity() if ci==co and s==1 else ConvBNAct(ci,co,1,s)
        self.residual=spec.residual and ci==co and s==1
    def forward(self,x):
        y=self.op(x); return y+x if self.residual else y

class EANASNet(nn.Module):
    def __init__(self,arch:Architecture,num_classes=10):
        super().__init__(); self.arch=arch
        self.stem=ConvBNAct(3,arch.stem_channels)
        ci=arch.stem_channels; blocks=[]
        for b in arch.blocks:
            blocks.append(SearchBlock(ci,b)); ci=b.channels
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d(1); self.classifier=nn.Linear(ci,num_classes)
    def forward(self,x): return self.classifier(self.pool(self.features(self.stem(x))).flatten(1))

def arch_a5():
    return Architecture((BlockSpec('dwconv3',32),BlockSpec('dwconv3',64,2),BlockSpec('attention',64),BlockSpec('dwconv3',128,2),BlockSpec('conv3',128,1,True)))

def random_arch(rng:random.Random):
    ch=32; bs=[]
    for i in range(5):
        op=rng.choice(OPS); s=rng.choice((1,1,2)) if i in (1,3) else 1
        co=rng.choice(WIDTHS) if i in (1,3) else ch
        bs.append(BlockSpec(op,co,s,rng.random()<.3));ch=co
    return Architecture(tuple(bs))

def mutate(arch,rng,prob=.2):
    bs=list(arch.blocks)
    for i,b in enumerate(bs):
        if rng.random()<prob:
            bs[i]=BlockSpec(rng.choice(OPS),b.channels,b.stride,rng.random()<.3)
    if tuple(bs)==arch.blocks:
        i=rng.randrange(5);b=bs[i];bs[i]=BlockSpec(rng.choice([x for x in OPS if x!=b.op]),b.channels,b.stride,b.residual)
    return Architecture(tuple(bs),arch.stem_channels)

def crossover(a,b,rng):
    k=rng.randint(1,4); return Architecture(a.blocks[:k]+b.blocks[k:],a.stem_channels)

def count_parameters(model): return sum(p.numel() for p in model.parameters())
