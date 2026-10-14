from pathlib import Path
import copy, json, random, time
import numpy as np
import torch
from torch import nn

def seed_all(seed,deterministic=True):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def accuracy(model,loader,device='cpu',max_batches=None):
    model.eval(); correct=total=0
    with torch.inference_mode():
        for i,(x,y) in enumerate(loader):
            if max_batches is not None and i>=max_batches:break
            pred=model(x.to(device)).argmax(1);correct+=(pred==y.to(device)).sum().item();total+=len(y)
    return correct/total if total else 0.0

def fit(model,train,val,epochs=10,lr=.025,weight_decay=.0005,momentum=.9,device='cpu',max_train_batches=None,max_val_batches=None,checkpoint=None):
    model.to(device); optim=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=max(1,epochs))
    best=-1; history=[];best_weights=None
    for epoch in range(epochs):
        model.train();loss_sum=0;n=0
        for j,(x,y) in enumerate(train):
            if max_train_batches is not None and j>=max_train_batches:break
            x=x.to(device);y=y.to(device);optim.zero_grad(set_to_none=True)
            logits=model(x);loss=nn.functional.cross_entropy(logits,y);loss.backward();optim.step()
            loss_sum+=loss.item()*len(x);n+=len(x)
        sched.step();val_acc=accuracy(model,val,device,max_val_batches)
        history.append(dict(epoch=epoch+1,loss=loss_sum/max(n,1),val_accuracy=val_acc))
        if val_acc>=best:best=val_acc;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_weights is not None: model.load_state_dict(best_weights)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'history':history,'best_val_accuracy':best},checkpoint)
    return {'val_accuracy':best,'history':history}
