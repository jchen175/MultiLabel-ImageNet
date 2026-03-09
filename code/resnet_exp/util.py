import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Literal
import os
import torch
import argparse

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.nn as nn
import json

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

class Ralloss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0, epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=False):
        super(Ralloss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.margin = 1.0
        self.lamb = lamb

    def forward(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2))
        los_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) ) * (self.lamb - x_sigmoid) * x_sigmoid ** 2 * (self.lamb - xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
    def __init__(self,
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 4.0,
                 clip: float = 0.05,
                 eps: float = 1e-8,
                 disable_torch_grad_focal_loss: bool = False,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, C) raw scores
        targets: (B, C) binary or soft targets in [0,1] (typical: {0,1})
        """
        # Sigmoid probabilities
        prob_pos = torch.sigmoid(logits)      # p
        prob_neg = 1.0 - prob_pos             # 1-p

        # Asymmetric clipping on negatives (downweight very easy negatives)
        if self.clip is not None and self.clip > 0:
            # (1-p) <- min( (1-p)+clip, 1 )
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        # Basic CE parts with numerical stability
        #   pos:   y * log(p)
        #   neg:  (1-y) * log(1-p_clipped)
        loss_pos = targets * torch.log(prob_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(prob_neg.clamp(min=self.eps))

        # Asymmetric focusing (like focal loss, but different gamma per side)
        if (self.gamma_pos > 0) or (self.gamma_neg > 0):
            if self.disable_torch_grad_focal_loss:
                prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)

            # (1 - p)^gamma_pos for positives, p^gamma_neg for negatives
            loss_pos *= torch.pow(1.0 - prob_pos, self.gamma_pos)
            loss_neg *= torch.pow(prob_pos,        self.gamma_neg)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(prev)

        # Combine and reduce (note the minus sign)
        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'

def smooth_targets_cap(y, pos_val=0.9, neg_val=0.05):
    """
    For positive entries: keep original if < pos_val, else cap at pos_val.
    For negatives (0): set to neg_val.

    y: (B,C) tensor in {0,1} or probabilities.
    """
    # start with all neg -> neg_val
    out = torch.full_like(y, neg_val)

    # for pos locations: cap at pos_val
    pos_mask = y > 0
    out[pos_mask] = torch.minimum(y[pos_mask], torch.tensor(pos_val, device=y.device, dtype=y.dtype))
    return out

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



Scheme = Literal["one_hot", "relabel_prob", "relabel_thresh"]

def build_filename_to_label(
        df: pd.DataFrame,
        num_classes: int = 1000,
        scheme: Scheme = "one_hot",
        *,
        label_cols: Optional[Iterable[str]] = ("m1_label", "m2_label", "m3_label"),
        prob_cols: Optional[Iterable[str]] = ("m1_prob", "m2_prob", "m3_prob"),
        include_gt: bool = False,
        threshold: float = 0.5,
        clip01: bool = True,
        filename_col: str = "filename",
        gt_col: str = "gt_label",
        verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Convert a dataframe of predictions into filename -> label-vector mapping.

    Parameters
    ----------
    df : DataFrame with columns:
        - filename
        - gt_label (int)
        - m{1,2,3}_label (class id; can be float or NaN)
        - m{1,2,3}_prob (probability; float or NaN)
    num_classes : size of label vector (default 1000)
    scheme :
        - "one_hot": vector is 1 only at gt_label (standard baseline)
        - "relabel_prob": fill vector with your proposal probs at their classes
        - "relabel_thresh": set 1 at proposal classes with prob >= threshold
    label_cols / prob_cols : which columns to use for proposals
    include_gt : if True, ensure ground-truth class is included
        - in "one_hot": no change (GT is already 1)
        - in "relabel_prob": y[gt] = max(y[gt], 1.0)
        - in "relabel_thresh": y[gt] = 1
    threshold : used only for "relabel_thresh"
    clip01 : clip final vector into [0,1] (useful for relabel_prob merges)
    filename_col, gt_col : column names

    Returns
    -------
    dict: filename -> np.float32[num_classes]
    """
    label_cols = list(label_cols or [])
    prob_cols  = list(prob_cols or [])

    if scheme in ("relabel_prob", "relabel_thresh") and len(label_cols) != len(prob_cols):
        raise ValueError("label_cols and prob_cols must have the same length.")

    out: Dict[str, np.ndarray] = {}

    # Defensive casts once to avoid per-row try/except
    def _to_int_or_none(x):
        try:
            if pd.isna(x):
                return None
            # some tables store as float like 391.0
            xi = int(x)
            return xi if 0 <= xi < num_classes else None
        except Exception:
            return None
    if verbose:
        print(f"building label map...")
        iterator = tqdm(df.iterrows(), total=len(df))
    else:
        iterator = df.iterrows()
    for _, row in iterator:
        fname = row[filename_col]
        y = np.zeros((num_classes,), dtype=np.float32)

        if scheme == "one_hot":
            gt = _to_int_or_none(row.get(gt_col, None))
            if gt is not None:
                y[gt] = 1.0

        elif scheme == "relabel_prob":
            for lc, pc in zip(label_cols, prob_cols):
                cls = _to_int_or_none(row.get(lc, None))
                prob = row.get(pc, np.nan)
                if cls is not None and pd.notna(prob):
                    # if multiple proposals hit same class, keep the max prob
                    y[cls] = max(float(prob), y[cls])

        elif scheme == "relabel_thresh":
            for lc, pc in zip(label_cols, prob_cols):
                cls = _to_int_or_none(row.get(lc, None))
                prob = row.get(pc, np.nan)
                if cls is not None and pd.notna(prob) and float(prob) >= threshold:
                    y[cls] = 1.0

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Optionally force-include GT
        if include_gt:
            gt = _to_int_or_none(row.get(gt_col, None))
            if gt is not None:
                if scheme == "relabel_prob":
                    y[gt] = max(1.0, y[gt])
                else:
                    y[gt] = 1.0

        if clip01:
            np.clip(y, 0.0, 1.0, out=y)

        out[fname] = y

    return out

def update_pred_gt(original_label_map, pred_gt_json, verbose=False):
    with open(pred_gt_json, "r") as f:
        data = json.load(f)
    if verbose:
        print("updating label map with predicted global labels...")
        iterable = tqdm(original_label_map.items(), total=len(original_label_map))
    else:
        iterable = original_label_map.items()
    for k, v in iterable:
        pred_l, pred_p = data[k]['pred'], data[k]['pred_prob']
        original_label_map[k][int(pred_l)] = max(original_label_map[k][int(pred_l)], float(pred_p))
    del data
    return original_label_map

def get_label_mapping_from_json(json_path, scheme, include_gt, threshold=0.5, verbose=False):
    if verbose:
        print(f"loading label map from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    label_mapping_from_json = {}
    if verbose:
        print("building label map from JSON...")
        iterable = tqdm(data.items(), total=len(data))
    else:
        iterable = data.items()
    for k, v in iterable:
        if scheme == 'one_hot':
            cur_label = np.zeros((1000,), dtype=np.float32)
            cur_label[int(v['gt'])] = 1.0
        else:
            cur_label = np.array(v['multilabel'], dtype=np.float32)
            if include_gt:
                cur_label[int(v['gt'])]=1.0
            if scheme == 'relabel_thresh':
                cur_label = (cur_label >= threshold).astype(np.float32)
        label_mapping_from_json[k] = cur_label
    del data
    return label_mapping_from_json

def get_label_mapping_from_pt(json_path, scheme, include_gt, threshold=0.5, verbose=False):
    data = torch.load(json_path, weights_only=False)
    label_mapping_from_json = {}
    if verbose:
        print("building label map from JSON...")
        iterable = tqdm(data.items(), total=len(data))
    else:
        iterable = data.items()
    for k, v in iterable:
        if scheme == 'one_hot':
            cur_label = np.zeros((1000,), dtype=np.float32)
            cur_label[int(v['gt'])] = 1.0
        else:
            cur_label = v['multilabel'].numpy().astype(np.float32)
            if include_gt:
                cur_label[int(v['gt'])]=1.0
            if scheme == 'relabel_thresh':
                cur_label = (cur_label >= threshold).astype(np.float32)
        label_mapping_from_json[k] = cur_label
    del data
    return label_mapping_from_json

class MultiLabelImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, label_mapping=None):
        super().__init__(root, transform=transform)
        self.label_mapping = label_mapping  # function to get extra labels from file path

    def __getitem__(self, index):
        # Get image and main label from ImageFolder
        image, main_label = super().__getitem__(index)
        # Get file path and use mapping function to get additional labels
        path, _ = self.samples[index]
        k = '/'.join(path.split('/')[-2:])
        multi_labels = []
        if self.label_mapping:
            multi_labels = self.label_mapping[k]
        return image, main_label, multi_labels


class ReaLValSet(Dataset):
    """
    add order information to calculate the real label acc
    """
    def __init__(self, root, transform=None, real_path='<DATA_ROOT>/dataset/imagenet/real.json'):
        self.real_path = real_path
        self.base_dataset = ImageFolder(root=root, transform=transform)
        self.real_label = self._load_real_label()

    def _load_real_label(self):
        with open(self.real_path) as f:
            real_labels = json.load(f)
        return real_labels

    def __len__(self):
        return len(self.base_dataset)

    def get_id(self, filename):
        basename = os.path.basename(filename)
        index = int(basename.split('_')[2])
        return index

    def _build_multilabel_vector(self, list_index):
        multi_label = torch.zeros(1000)
        for idx in list_index:
            multi_label[idx] = 1.0
        is_null = len(list_index) == 0
        return multi_label, is_null

    def __getitem__(self, index):
        item = self.base_dataset[index]
        idx = self.get_id(self.base_dataset.imgs[index][0])
        multi_label, is_null = self._build_multilabel_vector(self.real_label[idx-1])
        out = {
            'image': item[0],
            'label': item[1],
            'index': idx,
            'multi_label': multi_label,
            'is_null': is_null,
        }
        return out

