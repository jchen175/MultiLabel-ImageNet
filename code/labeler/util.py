import torch
from models.cls_head import get_cls_head

def load_state_dict_with_ddp_handling(model, state_dict, verbose=False):
    """
    Load state dict handling both DDP and non-DDP checkpoints.
    Strips 'module.' prefix if present and model doesn't expect it.
    """
    if state_dict is None:
        if verbose:
            print("Warning: Attempting to load None state_dict")
        return

    # Get the first key to check if state_dict has 'module.' prefix
    state_dict_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())

    if len(state_dict_keys) == 0:
        if verbose:
            print("Warning: Empty state_dict")
        return

    if len(model_keys) == 0:
        if verbose:
            print("Warning: Model has no parameters")
        return

    # Check if state_dict has 'module.' prefix but model doesn't expect it
    state_dict_has_module = state_dict_keys[0].startswith('module.')
    model_expects_module = model_keys[0].startswith('module.')

    if state_dict_has_module and not model_expects_module:
        # Strip 'module.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key  # Remove 'module.'
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        if verbose:
            print("Loaded state dict after stripping 'module.' prefix")
    elif not state_dict_has_module and model_expects_module:
        # Add 'module.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = f'module.{key}'
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        if verbose:
            print("Loaded state dict after adding 'module.' prefix")
    else:
        # No conversion needed
        model.load_state_dict(state_dict)
        if verbose:
            print("Loaded state dict directly (no prefix conversion needed)")


def get_masked_prediction(
        model,
        x: torch.Tensor,  # (B, 3, H, W)
        classification_head,
        *,
        patch_dim: int = 1,  # index of the patch dimension in (B, N_patch, C)
        mask=None,  # (B, H_mask, W_mask) or (B, N_patch)
        random_dropout: float = 0.0,  # fraction of *masked-in* patches to drop
        pool_first: bool = True,  # True: pool features -> head; False: head -> pool logits
        return_intermediates: bool = False  # optionally return pooled features / per-patch logits
):
    assert 0.0 <= random_dropout < 1.0, "`random_dropout` must be in [0, 1)"

    # 1) Extract patch features: (B, N_patch, C)
    with torch.no_grad():
        patch_feat = model.forward_features(x)['x_norm_patchtokens']
    B, N_patch, C = patch_feat.shape

    # 2) Build/validate mask over patches -> (B, N_patch) boolean
    if mask is None:
        mask_flat = torch.ones(B, N_patch, device=patch_feat.device, dtype=torch.bool)
    else:
        if mask.dim() == 3:  # (B, H_mask, W_mask)
            mask_flat = mask.flatten(1).bool()  # -> (B, N_patch)
        elif mask.dim() == 2:  # (B, N_patch)
            mask_flat = mask.bool()
        else:
            raise ValueError("`mask` must be (B, H, W) or (B, N_patch).")
        if mask_flat.shape != (B, N_patch):
            raise ValueError(f"Mask shape mismatch: got {mask_flat.shape}, expected {(B, N_patch)}")

    # 3) Randomly drop a subset of True entries (only among masked-in patches)
    if random_dropout > 0.0:
        keep_prob = 1.0 - random_dropout
        rand = torch.rand_like(mask_flat.float())
        dropout_mask = rand < keep_prob  # Bernoulli
        mask_flat = mask_flat & dropout_mask

        # Guarantee at least one patch per sample
        empty = (mask_flat.sum(dim=1) == 0)
        if empty.any():
            # fall back to all-patch averaging for those samples
            mask_flat[empty] = True

    mask_float = mask_flat.float().unsqueeze(-1)  # (B, N_patch, 1) for broadcasting
    n_kept = mask_float.sum(dim=patch_dim)  # (B, 1)

    extras = {'mask_flat': mask_flat}

    if pool_first:
        # 4A) Pool features -> (B, C)
        masked_feat = patch_feat * mask_float
        pooled_feat = masked_feat.sum(dim=patch_dim) / (n_kept + 1e-8)  # (B, C)

        # 5A) Classify pooled features -> (B, K)
        logits = classification_head(pooled_feat)  # assumes head: (B, C) -> (B, K)

        if return_intermediates:
            extras['pooled_feat'] = pooled_feat
        return (logits, extras) if return_intermediates else logits

    else:
        # 4B) Classify each patch first -> per-patch logits (B, N, K)
        # Many heads expect (B, C). We vectorize over patches:
        feats_2d = patch_feat.reshape(B * N_patch, C)  # (B*N, C)
        logits_2d = classification_head(feats_2d)  # (B*N, K)
        if logits_2d.dim() != 2:
            raise RuntimeError("classification_head must return 2D logits of shape (BN, K) for per-patch input.")
        K = logits_2d.shape[1]
        per_patch_logits = logits_2d.view(B, N_patch, K)  # (B, N, K)

        # 5B) Pool logits within mask -> (B, K)
        masked_logits = per_patch_logits * mask_float  # broadcast over last dim
        logits = masked_logits.sum(dim=patch_dim) / (n_kept + 1e-8)  # (B, K)

        if return_intermediates:
            extras['per_patch_logits'] = per_patch_logits
        return (logits, extras) if return_intermediates else logits


def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from keys in the state_dict.
    """
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def load_cls_head(ckpt, feat_dim=1024, out_dim=1000):
    head_state_dict = remove_module_prefix(ckpt['head_state_dict'])
    head = get_cls_head(
        arch=ckpt['args'].arch,
        input_dim=feat_dim,
        output_dim=out_dim,
        args=ckpt['args']
    )
    head.load_state_dict(head_state_dict)
    print(f"all keys loaded successfully")
    return head

def get_pooled_patch_features(
    model,
    x: torch.Tensor,                 # (B, 3, H, W)
    patch_dim: int = 1,              # index of the patch dimension in model output
    mask = None,            # (B, H_mask, W_mask) or (B, N_patch)
    random_dropout: float = 0.0,     # 0.0‑1.0  fraction of *masked‑in* patches to remove
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
        if mask.dim() == 3:                       # e.g. (B, H_mask, W_mask)
            mask_flat = mask.flatten(1).bool()           # -> (B, N_patch)
        else:
            mask_flat = mask.bool()               # already flat
        assert mask_flat.shape == (B, N_patch), "Mask shape mismatch"

    # 3) Randomly drop a subset of *True* entries -----------------------
    if random_dropout > 0.0:
        keep_prob = 1.0 - random_dropout
        rand = torch.rand_like(mask_flat.float())
        dropout_mask = rand < keep_prob          # Bernoulli
        mask_flat = mask_flat & dropout_mask

        # guarantee at least one patch per sample
        empty = (mask_flat.sum(dim=1) == 0)
        if empty.any():
            mask_flat[empty] = True              # fall back to all‑patch averaging

    mask_float = mask_flat.float().unsqueeze(-1)     # (B, N_patch, 1) for broadcasting

    # 4) Weighted average over remaining patches ------------------------
    masked_feat = patch_feat * mask_float
    n_kept      = mask_float.sum(dim=patch_dim)          # (B, 1)
    pooled      = masked_feat.sum(dim=patch_dim) / (n_kept + 1e-8)

    return pooled  # (B, C)