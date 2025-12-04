import torch
import torch.nn as nn
from monai.networks.nets import ResNetFeatures
from lightly.models.modules import SimCLRProjectionHead

class ResNet3d(nn.Module):
    def __init__(self, in_channels=11):
        super().__init__()
        self.resnet = ResNetFeatures('resnet50', pretrained=False, spatial_dims=3, in_channels=in_channels)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  
        self.projection_head = SimCLRProjectionHead()  
    
    def forward(self, x):
        out = self.resnet(x)  
        feature = self.gap(out[-1]).squeeze()  # (batch_size, 2048)

        if len(feature.shape) == 1:
            feature = feature.unsqueeze(0)
        projection = self.projection_head(feature) 

        return projection, feature