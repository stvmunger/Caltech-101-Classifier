import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DEVICE

def get_model(backbone='resnet18', pretrained=True):
    if backbone == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, NUM_CLASSES)
    elif backbone == 'alexnet':
        m = models.alexnet(pretrained=pretrained)
        in_f = m.classifier[6].in_features
        m.classifier[6] = nn.Linear(in_f, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return m.to(DEVICE)
