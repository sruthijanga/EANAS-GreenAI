#!/usr/bin/env python3
"""Export Arch-A5 for an ONNX-compatible device runtime."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from eanas.core import EANASNet,arch_a5,Architecture
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--architecture-json');p.add_argument('--classes',type=int,default=10);p.add_argument('--input-size',type=int,default=32);p.add_argument('--out',default='checkpoints/arch_a5.onnx');a=p.parse_args()
arch=Architecture.from_dict(__import__('json').loads(Path(a.architecture_json).read_text())) if a.architecture_json else arch_a5()
m=EANASNet(arch,a.classes).eval()
if a.checkpoint:
    obj=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(obj.get('model',obj))
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(m,torch.randn(1,3,a.input_size,a.input_size),a.out,input_names=['image'],output_names=['logits'],opset_version=17,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}})
print('Saved',a.out)
