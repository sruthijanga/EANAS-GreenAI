from torch import nn

def get_baseline(name,num_classes=10,pretrained=False):
    from torchvision import models
    name=name.lower().replace('-','').replace('_','')
    if name=='mobilenetv2':return models.mobilenet_v2(weights=None if not pretrained else models.MobileNet_V2_Weights.DEFAULT,num_classes=num_classes)
    if name=='efficientnetb0':return models.efficientnet_b0(weights=None if not pretrained else models.EfficientNet_B0_Weights.DEFAULT,num_classes=num_classes)
    if name in ('nasnetedge','dartscompact'):
        raise NotImplementedError(f'{name}: manuscript does not identify an exact architecture/genotype; supply a verified implementation rather than mislabeling an arbitrary network')
    raise ValueError(f'Unknown baseline: {name}')
