"""Dataset loading. torchvision import is lazy to allow use of core modules without it."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

CIFAR_MEAN=(.4914,.4822,.4465); CIFAR_STD=(.2470,.2435,.2616)
IMAGENET_MEAN=(.485,.456,.406); IMAGENET_STD=(.229,.224,.225)

def loaders(dataset='cifar10',root='data',batch_size=128,workers=2,seed=42,image_size=224,synthetic=False,train_samples=256,val_samples=64,download=True):
    if synthetic:
        g=torch.Generator().manual_seed(seed)
        nclass=1000 if dataset=='imagenet' else 10
        size=32 if dataset=='cifar10' else image_size
        x=torch.randn(train_samples+val_samples,3,size,size,generator=g)
        y=torch.randint(nclass,(len(x),),generator=g)
        t=TensorDataset(x[:train_samples],y[:train_samples]);v=TensorDataset(x[train_samples:],y[train_samples:]);test=v
    else:
        from torchvision import datasets, transforms as T
        if dataset=='cifar10':
            tr=T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(CIFAR_MEAN,CIFAR_STD)])
            ev=T.Compose([T.ToTensor(),T.Normalize(CIFAR_MEAN,CIFAR_STD)])
            # Independent training/evaluation transforms use identical underlying indices.
            train_ds=datasets.CIFAR10(root,train=True,transform=tr,download=download)
            val_ds=datasets.CIFAR10(root,train=True,transform=ev,download=False)
            test=datasets.CIFAR10(root,train=False,transform=ev,download=download)
            from collections import defaultdict
            groups=defaultdict(list)
            for idx,label in enumerate(train_ds.targets): groups[int(label)].append(idx)
            g=torch.Generator().manual_seed(seed);train_ids=[];val_ids=[]
            for label,indices in sorted(groups.items()):
                perm=torch.randperm(len(indices),generator=g).tolist()
                nval=int(.2*len(indices)); val_ids.extend(indices[j] for j in perm[:nval]);train_ids.extend(indices[j] for j in perm[nval:])
            t=Subset(train_ds,train_ids);v=Subset(val_ds,val_ids)
        elif dataset=='imagenet':
            root=Path(root)
            tr=T.Compose([T.RandomResizedCrop(image_size),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
            ev=T.Compose([T.Resize(round(image_size*256/224)),T.CenterCrop(image_size),T.ToTensor(),T.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
            if not (root/'train').exists() or not (root/'val').exists(): raise FileNotFoundError(f'ImageNet needs {root}/train and {root}/val class subfolders')
            t=datasets.ImageFolder(root/'train',transform=tr);v=datasets.ImageFolder(root/'val',transform=ev);test=v
        else: raise ValueError('MLPerf Tiny is benchmarked via scripts/run_mlperf.py; not a single ImageFolder dataset')
    gen=torch.Generator().manual_seed(seed)
    kwargs=dict(batch_size=batch_size,num_workers=workers,pin_memory=torch.cuda.is_available())
    return (DataLoader(t,shuffle=True,generator=gen,**kwargs),DataLoader(v,shuffle=False,**kwargs),DataLoader(test,shuffle=False,**kwargs))
