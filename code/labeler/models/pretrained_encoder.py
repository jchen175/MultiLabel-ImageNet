import os
import torch
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet50
from models.model_hook import DinoVitWrapper, Mocov3Wrapper, Dinov2VitWrapper, ResnetWrapper
from einops import rearrange

def get_sololearn_model(ckpt_root, ssl_method):
    """
     Load a pre-trained ResNet18 from solo-learn: https://github.com/vturrisi/solo-learn/tree/main?tab=readme-ov-file
    """
    from torchvision.models import resnet18
    all_methods = os.listdir(ckpt_root)
    ckpt_path = None
    for method in all_methods:
        if method.startswith(ssl_method):
            ckpt_path = os.path.join(ckpt_root, method, 'model.pth')
            print(f"Found checkpoint for {ssl_method} in {ckpt_path}: loading...")
            break
    assert ckpt_path is not None, f"Checkpoint not found for {ssl_method} in {ckpt_root}"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict']
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]

    model = resnet18()
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

    assert missing_keys == ['fc.weight', 'fc.bias'], f"Missing keys: {missing_keys}"
    model.fc = torch.nn.Identity()
    return model


def get_ssl_feature_extractor(ssl_method, ckpt_root='./checkpoints/pretrained'):
    if ssl_method == 'dinovit':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    elif ssl_method == 'dinores':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif ssl_method == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    elif ssl_method == 'mocov3vit':
        state_dict = torch.load(os.path.join(ckpt_root, 'mocov3/linear-vit-s-300ep.pth.tar'))['state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, num_classes=0
        )
        model.load_state_dict(new_state_dict, strict=False)
    elif ssl_method == 'mocov3res':
        model = resnet50(pretrained=False)
        state_dict = torch.load(os.path.join(ckpt_root, 'mocov3/r-50-1000ep.pth.tar'))['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the classification layer
            if k.startswith('module.base_encoder.'):
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.fc = torch.nn.Identity()
    elif ssl_method == 'btIN1k':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        model.fc = torch.nn.Identity()
    else:
        model = get_sololearn_model(ckpt_root, ssl_method)
    model.eval()
    return model


def get_hooked_model(ssl_method, ckpt_root='./checkpoints/pretrained', return_model_info=False, layer=None):
    model_dict = get_ssl_model_info(ssl_method, ckpt_root)
    model = model_dict.pop('model')
    if ssl_method == 'dinovit':
        model = DinoVitWrapper(model)
    elif ssl_method == 'dinov2':
        model = Dinov2VitWrapper(model)
    elif ssl_method == 'mocov3vit':
        model = Mocov3Wrapper(model)
    else:
        model = ResnetWrapper(model, layer_to_hook=layer)
    if return_model_info:
        return model, model_dict
    return model


def parse_ssl_method(ssl_method):
    method_specs = {
        'dinovit': ['dino', 'vits16', 'IN1k'],
        'dinores': ['dino', 'res50', 'IN1K'],
        'dinov2': ['dinov2', 'vits14', '142M'],
        'mocov3vit': ['mocov3', 'vits16', 'IN1k'],
        'mocov3res': ['mocov3', 'res50', 'IN1K'],
        'btIN1k': ['barlowtwins', 'res50', 'IN1K'],
        'btIN1h': ['barlowtwins', 'res18', 'IN1h'],
        'byolIN1h': ['byol', 'res18', 'IN1h'],
        'all4oneIN1h': ['all4one', 'res18', 'IN1h'],
    }
    assert ssl_method in method_specs, f"SSL method {ssl_method} not recognized."
    ssl_algorithm, arch, pretrain_set = method_specs[ssl_method]
    return pretrain_set, ssl_algorithm, arch


def test_model_out_dim(model, input_size=(1, 3, 224, 224)):
    """
    Test the output dimension of a model by passing a dummy input.
    """
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        output = model(dummy_input)
    return output.shape[-1]  # Return the last dimension which is the output feature size


def get_ssl_model_info(ssl_config, ckpt_root='./checkpoints/pretrained'):
    """
    Get the SSL model information including the model instance and output dimension.
    """
    pretrain_set, ssl_algorithm, arch = parse_ssl_method(ssl_config)
    model = get_ssl_feature_extractor(ssl_config, ckpt_root)
    output_dim = test_model_out_dim(model)
    return {
        'model': model,
        'arch': arch,
        'output_dim': output_dim,
        'pretrain_set': pretrain_set,
        'ssl_algorithm': ssl_algorithm,
    }


def get_dinov2_models(key):
    """
    Load DINOv2 models with registers.
    Returns:
        dict: A dictionary containing DINOv2 models.
    """
    assert key in ['vits', 'vitb', 'vitl', 'vitg']

    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{key}14_reg')
    size = model.forward_features(torch.randn(1, 3, 224, 224))['x_norm_patchtokens'].size()[-1]
    return model, size

class ViTLastLayerExtractor(torch.nn.Module):
    """
    Wraps a DINO-v2 ViT and exposes last-layer q / k / v or the
    normal patch tokens.  Output shape is always (B, T-5, embed_dim).
    """
    def __init__(self, base_vit, default="k", num_cls_reg_tokens=5):   # default pick if user does not specify
        super().__init__()
        assert default in {"q", "k", "v", "patch"}
        self.base = base_vit
        self.default = default
        self.num_cls_reg_tokens = num_cls_reg_tokens

        # ----- register one forward hook on *the very last* attention -----
        if getattr(self.base, "chunked_blocks", False):   # g-14 uses chunked_blocks
            self.last_block = self.base.blocks[-1][-1]
        else:
            self.last_block = self.base.blocks[-1]

        self._container = {}   # will hold q/k/v for the current forward pass

        def _hook(attn_mod, inputs, output):
            """
            inputs[0] : x  -> (B, T, embed_dim)
            Store q, k, v as (B, H, T, d) in self._container
            """
            x = inputs[0]
            B, T, C = x.shape
            qkv = attn_mod.qkv(x).reshape(
                B, T, 3, attn_mod.num_heads, C // attn_mod.num_heads
            ).permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
            self._container["q"], self._container["k"], self._container["v"] = qkv

        # keep the handle so the hook lives as long as the wrapper
        self._hook_handle = self.last_block.attn.register_forward_hook(_hook)

    # ---------------------------------------------------------------------
    @torch.no_grad()          # remove if you need gradients
    def forward(self, images, return_type=None):
        """
        return_type ∈ {"q", "k", "v", "patch"}; falls back to self.default.
        Output: (B, T-5, embed_dim)  where embed_dim = heads * head_dim
        """
        rtype = return_type or self.default
        assert rtype in {"q", "k", "v", "patch"}

        # run one pass – forward_features is enough, no classification head
        feats = self.base.forward_features(images)        # dict

        if rtype == "patch":
            x = feats["x_norm_patchtokens"]
            x = rearrange(x, "b t d -> b d t")
            return x

        # otherwise take what the hook cached
        tensor = self._container[rtype]                   # (B, H, T, d)
        tensor = tensor[:, :, self.num_cls_reg_tokens:, :]                      # remove first 5 (cls + reg) tokens
        tensor = rearrange(tensor, "b h t d -> b (h d) t")
        return tensor


def get_pooled_v_value(hooked_model, x, patch_dim=1, mask=None):
    v_value = hooked_model(x)  # B, C, N_patch
    v_value = rearrange(v_value, 'B C N_patch -> B N_patch C')  # (B, N_patch, C)

    if mask is not None:
        mask = mask.view(mask.shape[0], -1)  # B, N_patch
        mask = mask.unsqueeze(-1)  # B, N_patch, 1
        # Apply mask and get mean of only unmasked patches
        masked_v_value = v_value * mask
        # Sum masked values and divide by number of unmasked patches
        n_unmasked = mask.sum(dim=patch_dim)  # B, 1
        pooled_v_value = masked_v_value.sum(dim=patch_dim) / (n_unmasked + 1e-8)  # B, C
    else:
        # If no mask, average all patches
        pooled_v_value = v_value.mean(dim=patch_dim)  # B, C

    return pooled_v_value  # (B, C)


def get_pooled_patch_features(
        model,
        x: torch.Tensor,  # (B, 3, H, W)
        patch_dim: int = 1,  # index of the patch dimension in model output
        mask=None,  # (B, H_mask, W_mask) or (B, N_patch)
        random_dropout: float = 0.0,  # 0.0‑1.0  fraction of *masked‑in* patches to remove
):
    """
    Forward pass + masked / randomly‑dropped pooling.

    Returns
    -------
    pooled_patch_feat : torch.Tensor
        Shape (B, C)
    """
    assert 0.0 <= random_dropout < 1.0, "`random_dropout` must be in [0, 1)"

    # 1) Patch features --------------------------------------------------
    patch_feat = model(x, is_training=True)['x_norm_patchtokens']  # (B, N_patch, C)
    B, N_patch, C = patch_feat.shape

    # 2) Build a boolean mask over patches ------------------------------
    if mask is None:
        mask_flat = torch.ones(B, N_patch, device=patch_feat.device, dtype=torch.bool)
    else:
        if mask.dim() == 3:  # e.g. (B, H_mask, W_mask)
            mask_flat = mask.flatten(1).bool()  # -> (B, N_patch)
        else:
            mask_flat = mask.bool()  # already flat
        assert mask_flat.shape == (B, N_patch), "Mask shape mismatch"

    # 3) Randomly drop a subset of *True* entries -----------------------
    if random_dropout > 0.0:
        keep_prob = 1.0 - random_dropout
        rand = torch.rand_like(mask_flat.float())
        dropout_mask = rand < keep_prob  # Bernoulli
        mask_flat = mask_flat & dropout_mask

        # guarantee at least one patch per sample
        empty = (mask_flat.sum(dim=1) == 0)
        if empty.any():
            mask_flat[empty] = True  # fall back to all‑patch averaging

    mask_float = mask_flat.float().unsqueeze(-1)  # (B, N_patch, 1) for broadcasting

    # 4) Weighted average over remaining patches ------------------------
    masked_feat = patch_feat * mask_float
    n_kept = mask_float.sum(dim=patch_dim)  # (B, 1)
    pooled = masked_feat.sum(dim=patch_dim) / (n_kept + 1e-8)

    return pooled  # (B, C)


def get_dinov3_models(key, repo_dir='<dinov3_repo_path>'):
    dinov3_loc = {
        'vits16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vits16plus': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vitb16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vitl16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vith16plus': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vit7b16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_tiny': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_small': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_base': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_large': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
    }
    model = torch.hub.load(repo_dir, f'dinov3_{key}', source='local', weights=dinov3_loc[key])
    size = model.forward_features(torch.randn(1, 3, 224, 224))['x_norm_patchtokens'].size()[-1]
    return model, size


if __name__ == "__main__":

    for method in \
            ['dinovit', 'dinores', 'dinov2', 'mocov3vit', 'mocov3res', 'btIN1k', 'btIN1h', 'byolIN1h', 'all4oneIN1h']:
        try:
            model_info = get_ssl_model_info(method)
            print(f"Method: {method}, "
                  f"Output Dimension: {model_info['output_dim']}"
                  f", Architecture: {model_info['arch']}, "
                  f"Pretrain Set: {model_info['pretrain_set']}")
        except Exception as e:
            print(f"Failed to load model for {method}: {e}")
