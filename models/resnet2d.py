import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead
import torchvision.models as models

class ResNet2d(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None, num_classes=1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead()  
    
    def forward(self, x):
        feature = self.resnet(x).squeeze()  
        if len(feature.shape) == 1:
            feature = feature.unsqueeze(0)
        projection = self.projection_head(feature)  
        return projection, feature