import torch
from torch import nn
from torch.nn import functional as F
import einops
"""
This module provides wrapper classes for different backbone models for feature extraction purposes.
"""

class ResnetWrapper(nn.Module):
    def __init__(self, model, layer_to_hook='layer4'):
        super(ResnetWrapper, self).__init__()
        self.model = model
        self.intermediate_features = {}
        self.hook_handles = []

        # Register both layer3 and layer4 hooks
        self.layer_keys = ['layer3', 'layer4']
        self.register_hooks({
            'layer3': self.model.layer3[-1],
            'layer4': self.model.layer4[-1]
        })
        if layer_to_hook is None:
            layer_to_hook = 'layer4'
        self.layer_to_hook = layer_to_hook
        assert self.layer_to_hook in ['layer3', 'layer4', 'layer3_4']

    def register_hooks(self, layer_dict):
        """Register forward hooks on the specified layers."""
        for name, layer in layer_dict.items():
            handle = layer.register_forward_hook(self._hook_fn(name))
            self.hook_handles.append(handle)

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.intermediate_features[name] = output

        return hook

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def clear_intermediate_features(self):
        self.intermediate_features.clear()

    def forward(self, x):
        self.clear_intermediate_features()
        pooled_feature = self.model(x)

        feat3 = self.intermediate_features['layer3']  # shape (B, C3, H3, W3)
        feat4 = self.intermediate_features['layer4']  # shape (B, C4, H4, W4)

        if self.layer_to_hook == 'layer3':
            patch_feature = feat3
        elif self.layer_to_hook == 'layer4' or self.layer_to_hook is None:
            patch_feature = feat4
        elif self.layer_to_hook == 'layer3_4':
            # Upsample layer4 to match layer3 spatially
            if feat3.shape[2:] != feat4.shape[2:]:
                feat4_upsampled = F.interpolate(feat4, size=feat3.shape[2:], mode='nearest')
            else:
                feat4_upsampled = feat4

            # Concatenate features along channel dim
            patch_feature = torch.cat([feat3, feat4_upsampled], dim=1)  # shape: (B, C3+C4, H, W)

        # Flatten to patch-level representation
        patch_feature = einops.rearrange(patch_feature, 'b c h w -> b (h w) c')

        return {
            'pooled_feature': pooled_feature,
            'patch_feature': patch_feature
        }

class DinoVitWrapper(nn.Module):
    def __init__(self, model):
        super(DinoVitWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass
        last_feature = self.model.get_intermediate_layers(x, n=1)[0]
        pooled_feature = last_feature[:,0,:]
        patch_feature = last_feature[:,1:,:]
        output = {
            'pooled_feature': pooled_feature,
            'patch_feature': patch_feature,
        }
        return output


class Mocov3Wrapper(nn.Module):
    def __init__(self, model):
        super(Mocov3Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass
        last_feature = self.model.forward_features(x)
        pooled_feature = last_feature[:,0,:]
        patch_feature = last_feature[:,1:,:]
        output = {
            'pooled_feature': pooled_feature,
            'patch_feature': patch_feature,
        }
        return output

# should be working for dinov2/3 models
class Dinov2VitWrapper(nn.Module):
    def __init__(self, model):
        super(Dinov2VitWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass
        feature_dict = self.model.forward_features(x)
        patch_feature = feature_dict['x_norm_patchtokens']
        pooled_feature = self.model.head(feature_dict['x_norm_clstoken'])
        output = {
            'pooled_feature': pooled_feature,
            'patch_feature': patch_feature,
        }
        return output
