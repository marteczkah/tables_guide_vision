import torch
import torch.nn as nn
from resnet import ResNET, ResNET2D, TarteResNET
from collections import OrderedDict
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lightly.models.modules import SimCLRProjectionHead
from huggingface_hub import hf_hub_download
import os 
import json

class RegressionModel(nn.Module):
    def __init__(self, backbone_dim, bias, all, backbone="", mmcl=False, multimodal=False, freeze_backbone=False, dim=3):
        super().__init__()
        self.dim = dim
        self.get_model()
        if len(backbone) != 0:
            checkpoint = torch.load(backbone, map_location=torch.device('cpu'), weights_only=False)
            if mmcl:
                checkpoint = checkpoint['state_dict']
                if 'target_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("target_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('target_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                elif 'network.encoder.resnet.conv1.weight' in checkpoint.keys():
                    state_dict = {k.replace("network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                elif 'online_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("online_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('online_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                else:
                    state_dict = {k.replace('encoder_imaging.', ''): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('projector_imaging', 'projection_head'): v for k, v in state_dict.items()}
                _, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                if unexpected_keys:
                    print("Missing keys (not found in checkpoint):")
                    print(unexpected_keys)
            else:
                checkpoint = {k.replace('image_encoder.', ''): v for k, v in checkpoint.items()}
                missing_keys, un = self.backbone.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print('missing: ', missing_keys)
        if freeze_backbone:
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.multimodal = multimodal
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(backbone_dim, 1)
        )
        self.regressor[-1].bias.data[0] = bias
        self.all = all
    
    def forward(self, input):
        if self.multimodal:
            if self.all:
                input0, input1, input2 = input
                projection, representation = self.backbone(input0, input1, input2)
            else:
                input0, input1 = input
                projection, representation = self.backbone(input0, input1)
        else:
            projection, representation = self.backbone(input)
        return self.regressor(representation)
    
    def get_model(self):
        if self.dim == 2:
            self.backbone = ResNET2D()
        else:
            self.backbone = ResNET()

resnet_to_seq = {
    "conv1": "0",
    "bn1": "1",
    "relu": "2",
    "maxpool": "3",
    "layer1": "4",
    "layer2": "5",
    "layer3": "6",
    "layer4": "7",
    "avgpool": "8"
}

def remap_checkpoint_keys(checkpoint_state):
    new_state = OrderedDict()
    for k, v in checkpoint_state.items():
        if not k.startswith("resnet."):
            new_state[k] = v
            continue

        # remove prefix
        k_no_prefix = k[len("resnet."):]

        # split top-level module
        parts = k_no_prefix.split(".", 1)  # split first dot
        top_module = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if top_module in resnet_to_seq:
            new_key = f"encoder_imaging.{resnet_to_seq[top_module]}"
            if rest:
                new_key += f".{rest}"
            new_state[new_key] = v
        else:
            # any other keys, just keep as is
            new_state[k] = v
    return new_state

class MatryoshkaClassificationModel(nn.Module):
    def __init__(self, backbone_dim, num_classes, all=False, freeze_backbone=False, backbone="", multimodal=False, mmcl=False, dim=3):
        super().__init__()
        self.dim = dim
        self.backbone_dim = backbone_dim
        self.get_model()
        if len(backbone) != 0:
            checkpoint = torch.load(backbone, map_location=torch.device('cpu'), weights_only=False)
            if mmcl:
                checkpoint = checkpoint['state_dict']
                if 'target_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("target_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('target_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                elif 'network.encoder.resnet.conv1.weight' in checkpoint.keys():
                    state_dict = {k.replace("network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                elif 'online_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("online_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('online_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)  
                else:
                    print('this is MMCL')
                    state_dict = {k.replace('encoder_imaging.', 'resnet.'): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('projector_imaging', 'projection_head'): v for k, v in state_dict.items()}
                    # state_dict = remap_checkpoint_keys(state_dict)
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                if unexpected_keys:
                    print("Unexpected keys (not found in checkpoint):")
                    print(unexpected_keys)
                if missing_keys:
                    print("Missing keys (not found in checkpoint):")
                    print(missing_keys)
            else:
                # state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("projection_head")}
                checkpoint = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items()}
                checkpoint = {k.replace("projection_heads.", ""): v for k, v in checkpoint.items()}
                missing_keys, unexpected_keys = self.backbone.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print('Missing keys: ', missing_keys)
        if freeze_backbone:
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.multimodal = multimodal
        self.all = all
        self.heads = nn.ModuleDict({
            str(d): nn.Linear(d, num_classes) for d in backbone_dim
        })
    
    def forward(self, input):
        projection, representation = self.backbone(input)
        for p in projection:
            print(p.shape)
        # logits = self.head(representation)
        logits = {}
        for i, d in enumerate(self.backbone_dim):
            rep_d = projection[i]   # take first d dims
            logits[str(d)] = self.heads[str(d)](rep_d)
        return projection, logits
    
    def get_model(self):
        if self.dim == 2:
            self.backbone = ResNET2D()
        else:
            base_path = '/lustre/groups/iml/projects/marta/'
            cache_dir = os.path.join(base_path, "data/pretrained_weights")
            repo_id = 'inria-soda/tarte'
            # Load configs
            config_file = 'tarte_pretrained_configs.json'
            config_path = hf_hub_download(repo_id=repo_id, filename=config_file, cache_dir=cache_dir)
            with open(config_path) as f:
                configs = json.load(f)
            self.backbone = TarteResNET(config=configs, dim_projection=self.backbone_dim)

class ClassificationModel(nn.Module):
    def __init__(self, backbone_dim, num_classes, all=False, freeze_backbone=False, backbone="", multimodal=False, mmcl=False, dim=3):
        super().__init__()
        self.dim = dim
        self.get_model()
        if len(backbone) != 0:
            checkpoint = torch.load(backbone, map_location=torch.device('cpu'), weights_only=False)
            if mmcl:
                checkpoint = checkpoint['state_dict']
                if 'target_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("target_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('target_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                elif 'network.encoder.resnet.conv1.weight' in checkpoint.keys():
                    state_dict = {k.replace("network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                elif 'online_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("online_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('online_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)  
                else:
                    print('this is MMCL')
                    state_dict = {k.replace('encoder_imaging.', ''): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('projector_imaging', 'projection_head'): v for k, v in state_dict.items()}
                    # state_dict = remap_checkpoint_keys(state_dict)
                    missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                if unexpected_keys:
                    print("Unexpected keys (not found in checkpoint):")
                    print(unexpected_keys)
                if missing_keys:
                    print("Missing keys (not found in checkpoint):")
                    print(missing_keys)
            else:
                # state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("projection_head")}
                checkpoint = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items()}
                missing_keys, unexpected_keys = self.backbone.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print('Missing keys: ', missing_keys)
        if freeze_backbone:
            for _, param in self.backbone.named_parameters():
                param.requires_grad = False
        self.multimodal = multimodal
        self.all = all
        self.head = torch.nn.Sequential(
            torch.nn.Linear(backbone_dim, num_classes)
        )
    
    def forward(self, input):
        if self.multimodal:
            if self.all:
                input0, input1, input2 = input
                projection, representation = self.backbone(input0, input1, input2)
            else:
                input0, input1 = input
                projection, representation = self.backbone(input0, input1)
        else:
            projection, representation = self.backbone(input)
        logits = self.head(representation)
        return representation, logits
    
    def get_model(self):
        if self.dim == 2:
            self.backbone = ResNET2D()
        else:
            self.backbone = ResNET()

# class ClassificationModel(nn.Module):
#     def __init__(self, backbone, backbone_dim, num_classes, all, multimodal=False):
#         super().__init__()
#         self.backbone = backbone
#         self.multimodal = multimodal
#         self.all = all
#         self.head = torch.nn.Sequential(
#             torch.nn.Linear(backbone_dim, num_classes)
#         )
    
#     def forward(self, input):
#         if self.multimodal:
#             if self.all:
#                 input0, input1, input2 = input
#                 projection, representation = self.backbone(input0, input1, input2)
#             else:
#                 input0, input1 = input
#                 projection, representation = self.backbone(input0, input1)
#         else:
#             projection, representation = self.backbone(input)
#         logits = self.head(representation)
#         return logits

class DVMRegressionModel(nn.Module):
    def __init__(self, backbone_dim, bias, all, backbone="", mmcl=False, multimodal=False, freeze_backbone=False, dim=3):
        super().__init__()
        self.dim = dim
        backbone_dim = 2048
        self.create_imaging_model()
        if len(backbone) != 0:
            checkpoint = torch.load(backbone, map_location=torch.device('cpu'), weights_only=False)
            if mmcl:
                checkpoint = checkpoint['state_dict']
                if 'target_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("target_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('target_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                elif 'network.encoder.resnet.conv1.weight' in checkpoint.keys():
                    state_dict = {k.replace("network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                elif 'online_network.projector.model.1.weight' in checkpoint.keys():
                    state_dict = {k.replace("online_network.encoder.", ""): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('online_network.projector.', 'projection_head.'): v for k, v in state_dict.items()}
                else:
                    state_dict = {k.replace('encoder_imaging.', ''): v for k, v in checkpoint.items()}
                    state_dict = {k.replace('projector_imaging', 'projection_head'): v for k, v in state_dict.items()}
                missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                if unexpected_keys:
                    print("Unexpected keys (not found in checkpoint):")
                    print(unexpected_keys)
                if unexpected_keys:
                    print("Missing keys (not found in checkpoint):")
                    print(missing_keys)
            else:
                state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("projection_head")}
                state_dict = {k.replace("resnet.", ""): v for k, v in state_dict.items()}
                # state_dict = self.remap_state_dict_keys(state_dict)
                # self.encoder.load_state_dict(state_dict, strict=False)
                load_info = self.encoder.load_state_dict(state_dict, strict=False)
                if load_info.missing_keys:
                    print("Missing keys:", load_info.missing_keys)
        if freeze_backbone:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
        self.multimodal = multimodal
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(backbone_dim, 1)
        )
        self.regressor[-1].bias.data[0] = bias
        self.all = all
    
    def forward(self, input):
        representation = self.encoder(input)
        return self.regressor(representation.squeeze())
    
    def create_imaging_model(self):
        model = models.resnet50(pretrained=False, num_classes=100)
        self.pooled_dim = 2048
        self.encoder = nn.Sequential(*list(model.children())[:-1])

class DVMClassificationModel(nn.Module):
    def __init__(self, backbone_path, num_classes, freeze_backbone=False, all=False, multimodal=False, mmcl=False):
        super().__init__()
        backbone_dim = 2048
        self.create_imaging_model()
        # resnet = models.resnet50(weights=None, num_classes=1)
        # self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # state_dict = torch.load(backbone_path, weights_only=False)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("projector_tabular")}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("projector_imaging")}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("encoder_tabular")}
        # state_dict = {k.replace("encoder_imaging.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("layer", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("encoder_imaging.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("encoder_imaging.", ""): v for k, v in state_dict.items()}
        if len(backbone_path)> 0:
            checkpoint = torch.load(backbone_path, weights_only=False)
            if mmcl:
                original_args = checkpoint['hyper_parameters']
                state_dict = checkpoint['state_dict']
                self.pooled_dim = 2048 if original_args['model']=='resnet50' else 512

                self.encoder_name = 'encoder_imaging.'
                
                state_dict_encoder = {}
                for k in list(state_dict.keys()):
                    if k.startswith(self.encoder_name) and not 'projection_head' in k and not 'prototypes' in k:
                        state_dict_encoder[k[len(self.encoder_name):]] = state_dict[k]

                state_dict_encoder = self.remap_state_dict_keys(state_dict_encoder)
                self.encoder.load_state_dict(state_dict_encoder, strict=True)
            else:
                self.create_imaging_model()
                state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("projection_head")}
                state_dict = {k.replace("resnet.", ""): v for k, v in state_dict.items()}
                # state_dict = self.remap_state_dict_keys(state_dict)
                # self.encoder.load_state_dict(state_dict, strict=False)
                load_info = self.encoder.load_state_dict(state_dict, strict=False)
                if load_info.missing_keys:
                    print("Missing keys:", load_info.missing_keys)
        
        if freeze_backbone:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False


        self.multimodal = multimodal
        self.all = all
        self.head = torch.nn.Sequential(
            torch.nn.Linear(backbone_dim, num_classes)
        )
    
    def forward(self, input):
        representation = self.encoder(input)
        logits = self.head(representation.squeeze())
        return logits

    def create_imaging_model(self):
        model = models.resnet50(pretrained=False, num_classes=100)
        self.pooled_dim = 2048
        self.encoder = nn.Sequential(*list(model.children())[:-1])
    
    def remap_key(self, key):
        if key.startswith("conv1."):
            return key.replace("conv1", "0", 1)
        elif key.startswith("bn1."):
            return key.replace("bn1", "1", 1)
        elif key.startswith("layer1."):
            return key.replace("layer1", "4", 1)
        elif key.startswith("layer2."):
            return key.replace("layer2", "5", 1)
        elif key.startswith("layer3."):
            return key.replace("layer3", "6", 1)
        elif key.startswith("layer4."):
            return key.replace("layer4", "7", 1)
        elif key.startswith("fc."):
            return key.replace("fc", "8", 1)
        else:
            return key  # or raise an error

    def remap_state_dict_keys(self, old_state_dict):
        return {self.remap_key(k): v for k, v in old_state_dict.items()}
    
class ClassificationModelCombined(nn.Module):
    def __init__(self, backbone_la, backbone_sa, num_classes, fusion_method, multimodal_embedding_dim=2048):
        super().__init__()
        self.backbone_sa = backbone_sa
        self.backbone_la = backbone_la
        self.fusion_method = fusion_method
        self.combine = torch.nn.Linear(multimodal_embedding_dim*2, multimodal_embedding_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(multimodal_embedding_dim, num_classes)
        )

    def forward(self, la, sa):
        _, representation_sa = self.backbone_sa(sa)
        _, representation_la = self.backbone_la(la)
        # representation = torch.max(representation_sa, representation_la)
        if self.fusion_method == 'CONCAT':
            x = torch.cat([representation_la, representation_sa], dim=1)
        elif self.fusion_method == 'MAX':
            x = torch.stack([representation_la, representation_sa], dim=1)
            x, _ = torch.max(x, dim=1)
        elif self.fusion_method == 'LINEAR':
            x = torch.cat([representation_la, representation_sa], dim=1)
            x = self.combine(x)
        logits = self.head(x)
        return logits

class OutputHead(nn.Module):
    def __init__(self, num_classes, embedding_dim=2048):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, feature):
        logits = self.head(feature)
        return logits

class TabularLanguageEncoder(nn.Module):
    def __init__(self, language_model, break_token_id, tokenizer_lenght, hidden_dim=1024, num_heads=4, num_layers=2, projection_head=False):
        super().__init__()
        self.lang_model = AutoModel.from_pretrained(language_model)
        self.lang_model.resize_token_embeddings(tokenizer_lenght)
        # Freeze all parameters
        for param in self.lang_model.parameters():
            param.requires_grad = False

        # Unfreeze only the token embedding layer
        for param in self.lang_model.get_input_embeddings().parameters():
            param.requires_grad = True
        self.break_token_id = break_token_id
        self.hidden_dim = hidden_dim

        # Feature-level CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        trunc_normal_(self.cls_token, std=.02)

        # Feature-level transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.feature_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.initialize_imaging_model()
        self.projection_head = projection_head
        if self.projection_head:
            self.projection_head_tabular = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

    def initialize_imaging_model(self):
        self.image_encoder = ResNET()
    
    def make_feature_embeddings(self, input_tokens, token_embeddings):
        """
        input_tokens: [batch_size, seq_len]  # token IDs including break tokens
        token_embeddings: [batch_size, seq_len, hidden_dim]
        returns: [batch_size, num_features, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = token_embeddings.size()
        feature_embeddings = []

        for b in range(batch_size):
            embeddings_b = token_embeddings[b]  # [seq_len, hidden_dim]
            tokens_b = input_tokens[b]          # [seq_len]

            # Find indices of break tokens
            break_positions = (tokens_b == self.break_token_id).nonzero(as_tuple=False).squeeze(-1)

            # Add start and end indices
            # splits = [0] + (break_positions + 1).tolist() + [seq_len]
            splits = [0] + (break_positions + 1).tolist() 

            # Compute mean embeddings per feature
            features_b = []
            for i in range(len(splits)-1):
                start, end = splits[i], splits[i+1]
                if start >= end:
                    continue
                feature_emb = embeddings_b[start:end].mean(dim=0)  # [hidden_dim]
                features_b.append(feature_emb)
            features_b = torch.stack(features_b, dim=0)  # [num_features, hidden_dim]
            feature_embeddings.append(features_b)

        return torch.stack(feature_embeddings, dim=0)

    
    def forward(self, input_tokens, feature_mask, img):
        vocab_size = self.lang_model.get_input_embeddings().weight.size(0)

        if (input_tokens < 0).any() or (input_tokens >= vocab_size).any():
            bad = (input_tokens < 0) | (input_tokens >= vocab_size)
            idx = bad.nonzero(as_tuple=False)[0]
            b, p = int(idx[0]), int(idx[1])
            raise ValueError(
                f"Invalid token id {int(input_tokens[b,p].item())} "
                f"at batch {b}, pos {p}, vocab_size={vocab_size}"
            )
        # tabular forward
        token_embeddings = self.lang_model.get_input_embeddings()(input_tokens)
        feature_embeddings = self.make_feature_embeddings(input_tokens, token_embeddings)
        transformer_out = self.feature_transformer(feature_embeddings,
                                                   src_key_padding_mask=feature_mask)
        cls_embedding = transformer_out[:, 0, :]  # [B, hidden_dim]
        if self.projection_head:
            cls_embedding = self.projection_head_tabular(cls_embedding)
        # image forward
        img_proj, _ = self.image_encoder(img)
        return feature_embeddings, transformer_out, cls_embedding, img_proj
        
