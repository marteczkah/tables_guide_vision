import torch
import torch.nn as nn
from resnet2d import ResNet2d
from resnet3d import ResNet3d

class RegressionModel(nn.Module):
    def __init__(self, backbone_dim, bias, all, backbone_path="", freeze_backbone=False, dim=3):
        super().__init__()
        self.dim = dim
        self.get_model()

        if len(backbone_path) != 0:
            checkpoint = torch.load(backbone_path, map_location=torch.device('cpu'), weights_only=False)
            checkpoint = {k.replace('image_encoder.', ''): v for k, v in checkpoint.items()}
            missing_keys, _ = self.backbone.load_state_dict(checkpoint, strict=False)
            if missing_keys:
                print('missing: ', missing_keys)

        if freeze_backbone:
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(backbone_dim, 1)
        )
        self.regressor[-1].bias.data[0] = bias
        self.all = all
    
    def forward(self, input):
        _, representation = self.backbone(input)
        return self.regressor(representation)
    
    def get_model(self):
        if self.dim == 2:
            self.backbone = ResNet2d()
        else:
            self.backbone = ResNet3d()