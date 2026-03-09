"""
Evaluation script for multi-label ImageNet classification.
Computes comprehensive metrics including mAP, precision, recall, F1, subset accuracy, and IoU.
Supports stratified evaluation by ground-truth label count buckets.
"""

import argparse
import os
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet50, resnet101
import timm
from timm import create_model
from timm.models import create_model as models_create_model
from tqdm import tqdm
from collections import OrderedDict
# handle different model versions; defined as in deit3 repo; git clone https://github.com/facebookresearch/deit.git
import models as models_v1
import models_v2

from transformers import AutoConfig, AutoModel, AutoModelForImageClassification


class HFClassifier(nn.Module):
    """
    Wraps a HF *ForImageClassification model so forward(x) returns logits.
    """
    def __init__(self, model_for_cls: nn.Module):
        super().__init__()
        self.backbone = model_for_cls  # e.g., ViTForImageClassification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is pixel_values, shape [B,3,H,W]; you handle transforms externally
        return self.backbone(pixel_values=x).logits


class HFBackboneWithHead(nn.Module):
    """
    Uses a HF base vision backbone (e.g., ViTModel) and adds a new Linear head.
    Keeps forward(x) -> logits.
    """
    def __init__(self, backbone: nn.Module, in_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.new_head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x, return_dict=True)
        # Prefer pooled output when available; otherwise use CLS token
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        else:
            # last_hidden_state: [B, seq_len, hidden]
            feats = out.last_hidden_state[:, 0]
        return self.new_head(feats)


def _get_classifier_module(hf_model: nn.Module) -> Optional[nn.Module]:
    """
    Best-effort grab of the classifier layer on common HF image models.
    """
    for attr in ["classifier", "score", "fc", "head", "pre_logits"]:  # most models use 'classifier'
        if hasattr(hf_model, attr) and isinstance(getattr(hf_model, attr), nn.Module):
            return getattr(hf_model, attr)
    return None


def _reset_hf_classifier(hf_model: nn.Module, num_classes: int) -> int:
    """
    Replace the HF model's classifier with a new Linear of size (in_features -> num_classes).
    Returns in_features used for the new head.
    """
    clf = _get_classifier_module(hf_model)
    if clf is None:
        raise ValueError("Could not locate classifier module on the HF model.")
    # Handle simple Linear classifier (most vision models on HF)
    if isinstance(clf, nn.Linear):
        in_features = clf.in_features
        new_clf = nn.Linear(in_features, num_classes)
        # put it back on the same attribute name
        for attr in ["classifier", "score", "fc", "head"]:
            if hasattr(hf_model, attr) and getattr(hf_model, attr) is clf:
                setattr(hf_model, attr, new_clf)
                break
    else:
        # Fall back: try to find the first Linear inside the classifier module
        linear_layers = [m for m in clf.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ValueError("Classifier found but no Linear layer to reset.")
        in_features = linear_layers[-1].in_features  # assume last linear is the logits layer
        # Replace entire classifier block with a plain Linear to num_classes
        for attr in ["classifier", "score", "fc", "head"]:
            if hasattr(hf_model, attr) and getattr(hf_model, attr) is clf:
                setattr(hf_model, attr, nn.Linear(in_features, num_classes))
                break

    # Keep config in sync
    if hasattr(hf_model, "config"):
        hf_model.config.num_labels = num_classes
        # You can also set id2label/label2id if you have them

    return in_features


# ---------- Your original control flow, adapted ----------

def build_model(args, num_classes: int):
    """
    args.arch: Hugging Face model id (e.g., 'google/vit-base-patch16-224')
    args.attach_head: bool
    """
    model_id = args.arch
    cfg = AutoConfig.from_pretrained(model_id)

    if getattr(args, "attach_head", False):
        # Build a backbone (no classification head) + new Linear head
        # AutoModel gives the "base" model (e.g., ViTModel / DeiTModel)
        backbone = AutoModel.from_pretrained(model_id, config=cfg)
        # hidden size is the feature dim for CLS/pooler
        in_dim = getattr(cfg, "hidden_size", None)
        if in_dim is None:
            # Fallback: try common names
            for name in ["embed_dim", "vision_config", "projection_dim"]:
                if hasattr(cfg, name):
                    maybe = getattr(cfg, name)
                    in_dim = getattr(maybe, "hidden_size", maybe) if hasattr(maybe, "hidden_size") else maybe
                    if isinstance(in_dim, int):
                        break
        if not isinstance(in_dim, int):
            raise ValueError("Could not determine backbone feature dimension for head attachment.")

        model = HFBackboneWithHead(backbone=backbone, in_dim=in_dim, num_classes=num_classes)

    else:
        # Use the classification variant and reset its classifier to num_classes
        base = AutoModelForImageClassification.from_pretrained(model_id, config=cfg)
        # If the current number of labels already matches, no change; else reset
        current = getattr(base.config, "num_labels", None)
        if current != num_classes:
            _reset_hf_classifier(base, num_classes=num_classes)
        model = HFClassifier(base)

    return model


# ============================================================================
# Constants
# ============================================================================

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMNET_STD = torch.tensor([0.229, 0.224, 0.225])

# Default bucket scheme: (1, 2, 3, >=4)
BUCKET_KEYS_DEFAULT = [
    ("1", "cnt1"),
    ("2", "cnt2"),
    ("3", "cnt3"),
    (">=4", "cnt4p")
]

# Simple bucket scheme: (1, 2, >=3) - better for datasets with few multilabel instances
BUCKET_KEYS_SIMPLE = [
    ("1", "cnt1"),
    ("2", "cnt2"),
    (">=3", "cnt3p")
]

# For backward compatibility
BUCKET_KEYS = BUCKET_KEYS_DEFAULT

class ViTGAPHead(nn.Module):
    def __init__(self, num_classes, model_key, img_size=224, drop_path_rate=0.2, drop_rate=0.1):
        super().__init__()
        self.backbone = create_model(model_key, pretrained=False, num_classes=0, img_size=img_size, drop_path_rate=drop_path_rate, drop_rate=drop_rate)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward_features(self, x):
        feats = self.backbone.forward_features(x)  # shape [B, N, C]
        gap = feats.mean(dim=1)  # [B, C]
        return gap

    def forward(self, x):
        z = self.forward_features(x)
        return self.fc(z)

# ============================================================================
# Dataset Classes
# ============================================================================

class ImageNetV2Dataset(Dataset):
    """ImageNet-V2 dataset with multi-label annotations."""
    
    def __init__(
        self,
        root: str,
        annotation_file_path: str,
        transform: Optional[transforms.Compose] = None
    ):
        self.base_dataset = ImageFolder(root, transform=transform)
        self._build_multilabel(annotation_file_path)

    def _build_multilabel(self, annotation_file_path: str):
        with open(annotation_file_path, 'r') as f:
            annotation_file = json.load(f)
        
        self.image_order = {}
        self.multilabel = [[] for _ in range(len(self.base_dataset))]
        
        for i, (k, v) in enumerate(annotation_file.items()):
            self.image_order[k] = i
            self.multilabel[i] = [int(v_) for v_ in v]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        image_filename = self.base_dataset.imgs[idx][0]
        index = self.image_order[image_filename.split('/')[-1]]
        label = int(image_filename.split('/')[-2])
        
        return {
            "image": image,
            "label": label,
            "index": index,
        }


class ReaLValSet(Dataset):
    """ImageNet validation set with index tracking for ReaL labels."""
    
    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None
    ):
        self.base_dataset = ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.base_dataset)

    def get_id(self, filename: str) -> int:
        """Extract index from filename (format: ILSVRC2012_val_00000001.JPEG)."""
        basename = os.path.basename(filename)
        index = int(basename.split('_')[2].split('.')[0])
        return index

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        filename = self.base_dataset.imgs[index][0]
        
        return {
            'image': image,
            'label': label,
            'index': self.get_id(filename),
            'name': os.path.basename(filename)
        }


# ============================================================================
# Transform Functions
# ============================================================================

def imagenet_transform(
    size: int = 224,
    mean: torch.Tensor = IMNET_MEAN,
    std: torch.Tensor = IMNET_STD,
    split: str = 'val',
    center_crop: bool = False,
    rand_aug: bool = False
) -> transforms.Compose:
    """Create ImageNet data transforms."""
    assert split in ['train', 'val'], f"Invalid split: {split}"
    
    trans_list = []
    
    if split == 'val':
        if not center_crop:
            trans_list.append(transforms.Resize((size, size)))
        else:
            resize_size = int(256 / 224 * size)
            trans_list.extend([
                transforms.Resize(resize_size),
                transforms.CenterCrop(size)
            ])
    elif split == 'train':
        trans_list.append(transforms.RandomResizedCrop((size, size)))
        trans_list.append(transforms.RandomHorizontalFlip())
        if rand_aug:
            trans_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    
    trans_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transforms.Compose(trans_list)


# ============================================================================
# Model Loading
# ============================================================================

def load_weights_remove_module(checkpoint: dict, model: torch.nn.Module) -> torch.nn.Module:
    """Load weights from checkpoint, removing 'module.' prefix if present."""
    new_ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            k = k[7:]
        new_ckpt[k] = v
    model.load_state_dict(new_ckpt)
    return model


def load_model(checkpoint_path: str = None, device: str = 'cuda:0', args=None) -> torch.nn.Module:
    """Load ResNet-50 model from checkpoint."""
    transforms = None
    if args.arch == 'resnet50':
        model = resnet50(weights=None, num_classes=1000)
    elif args.arch == 'resnet101':
        if not args.finetune:
            model = resnet101(weights=None, num_classes=1000)
        else:
            model = resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2, num_classes=1000)
    elif 'vit' in args.arch and args.finetune:
        model = create_model(f"timm/deit3_{args.arch.split('_')[1]}_patch16_{args.size}.fb_in1k", pretrained=True)
    elif 'vits' in args.arch and not args.finetune:
        model = ViTGAPHead(num_classes=1000, model_key=f'{args.arch}_patch16_{args.size}', img_size=args.size)
    elif 'deit' in args.arch and 'timm' not in args.arch:
        model = models_create_model(
            args.arch,
            pretrained=False,
            num_classes=1000,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.size
        )
    elif args.use_huggingface:
        print('Using huggingface model')
        model = build_model(args, num_classes=1000)

    else:
        # timm model
        print(f'using timm for model creation: {args.arch}')
        model = create_model(args.arch, pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        out_dim = model.get_classifier().out_features
        if out_dim != 1000:
            print(f"num_classes of loaded model is {out_dim}, modifying to 1000.")
            if args.attach_head:
                model = nn.Sequential(OrderedDict([
                    ("backbone", model),
                    ("new_head", nn.Linear(out_dim, 1000)),
                ]))
            else:
                # modify the existing head
                model.reset_classifier(num_classes=1000)


    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, weights_only=False)
        loaded = False

        # Try different checkpoint formats
        for key in ['model', 'ema_dict', 'model_dict', 'model_state_dict']:
            if key in ckpt:
                model = load_weights_remove_module(ckpt[key], model)
                loaded = True
                break
        if not loaded:
            # If no standard key found, try loading directly
            model = load_weights_remove_module(ckpt, model)
    
    model.eval()
    model.to(device)
    return model, transforms


# ============================================================================
# Threshold Calibration
# ============================================================================

def calibrate_thresholds_by_count(
    logits: np.ndarray,
    gt_counts: Dict[int, int]
) -> np.ndarray:
    """
    Calibrate per-class thresholds so predicted positives match GT counts.
    
    Args:
        logits: (N, C) array of raw logits
        gt_counts: dict {class_idx: count} of ground truth class occurrences
    
    Returns:
        thresholds: (C,) array of per-class thresholds
    """
    N, C = logits.shape
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    
    thresholds = np.zeros(C)
    for c in range(C):
        desired = gt_counts.get(c, 0)
        if desired == 0:
            thresholds[c] = 1.0  # never predict positives
            continue
        
        # Sort probabilities descending
        sorted_probs = np.sort(probs[:, c])[::-1]
        
        # Pick cutoff at rank=desired
        if desired < len(sorted_probs):
            thresholds[c] = sorted_probs[desired - 1]
        else:
            thresholds[c] = sorted_probs[-1]
    
    return thresholds




# ============================================================================
# Core Metric Computation Functions
# ============================================================================

def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 if denominator is 0."""
    return numerator / denominator if denominator != 0 else 0.0


def _bucket_of(k: int, bucket_scheme: str = 'default') -> str:
    """
    Determine bucket based on GT label count.
    
    Args:
        k: Number of GT labels
        bucket_scheme: Either 'default' (1,2,3,>=4) or 'simple' (1,2,>=3)
    
    Returns:
        Bucket key as string
    """
    if k <= 0:
        return "0"
    elif k == 1:
        return "1"
    elif k == 2:
        return "2"
    elif k == 3:
        if bucket_scheme == 'simple':
            return ">=3"
        else:
            return "3"
    else:  # k >= 4
        if bucket_scheme == 'simple':
            return ">=3"
        else:
            return ">=4"


def _filter_nonempty_gt(
    gts: List[List[int]],
    preds: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """Keep only samples with at least one GT label."""
    keep_idx = [i for i, y in enumerate(gts) if len(y) > 0]
    gts_filtered = [gts[i] for i in keep_idx]
    preds_filtered = [preds[i] for i in keep_idx]
    return gts_filtered, preds_filtered, keep_idx


def _tp_fp_fn_for_sample(
    gt: List[int],
    pred: List[int]
) -> Tuple[int, int, int]:
    """Compute TP, FP, FN for a single sample (set-based)."""
    gt_set, pred_set = set(gt), set(pred)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    return tp, fp, fn


def compute_average_precision(
    gt_labels: List[int],
    pred_scores: np.ndarray,
    num_classes: int
) -> float:
    """
    Compute Average Precision (AP) for a single sample.
    
    Args:
        gt_labels: List of ground truth class indices
        pred_scores: (C,) array of prediction scores/probabilities for all classes
        num_classes: Total number of classes
    
    Returns:
        Average Precision score
    """
    if len(gt_labels) == 0:
        return 0.0
    
    # Create binary GT vector
    gt_binary = np.zeros(num_classes, dtype=np.int32)
    for label in gt_labels:
        if 0 <= label < num_classes:
            gt_binary[label] = 1
    
    # Sort predictions in descending order
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_gt = gt_binary[sorted_indices]
    
    # Compute precision at each position where we have a true positive
    num_positives = len(gt_labels)
    precisions = []
    num_correct = 0
    
    for i, is_positive in enumerate(sorted_gt):
        if is_positive == 1:
            num_correct += 1
            precision = num_correct / (i + 1)
            precisions.append(precision)
    
    # Average precision is the mean of precisions at recall points
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)


# ============================================================================
# Main Metrics Computation
# ============================================================================

def compute_multilabel_metrics(
    gts: List[List[int]],
    preds: List[List[int]],
    pred_scores: Optional[np.ndarray] = None,
    num_classes: int = 1000,
    bucket_scheme: str = 'default'
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-label metrics.
    
    Args:
        gts: List of ground truth label lists
        preds: List of predicted label lists
        pred_scores: Optional (N, C) array of prediction scores for mAP computation
        num_classes: Number of classes
        bucket_scheme: Either 'default' (1,2,3,>=4) or 'simple' (1,2,>=3)
    
    Returns:
        Dictionary containing overall and bucketed metrics
    """
    assert len(gts) == len(preds), "gts and preds must have same length"
    
    # Filter out empty-GT samples
    gts, preds, kept_indices = _filter_nonempty_gt(gts, preds)
    N = len(gts)
    
    if N == 0:
        return {"note": "No samples with non-empty GT after filtering."}
    
    # Filter pred_scores to match filtered samples
    if pred_scores is not None:
        pred_scores = pred_scores[kept_indices]
    
    # Initialize accumulators
    micro_tp = micro_fp = micro_fn = 0
    subset_acc_hits = 0
    iou_sum = 0.0
    ap_sum = 0.0
    
    # Per-class accumulators
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_has_gt = defaultdict(bool)
    
    # Bucketed accumulators - use appropriate bucket keys
    bucket_keys = BUCKET_KEYS_SIMPLE if bucket_scheme == 'simple' else BUCKET_KEYS_DEFAULT
    bucket_names = [b[0] for b in bucket_keys]
    buckets = {
        b: {
            "tp": 0, "fp": 0, "fn": 0,
            "subset_hits": 0, "count": 0,
            "iou_sum": 0.0, "ap_sum": 0.0
        }
        for b in bucket_names
    }
    
    # Iterate through samples
    for idx, (y, p) in enumerate(zip(gts, preds)):
        # TP/FP/FN for micro metrics
        tp, fp, fn = _tp_fp_fn_for_sample(y, p)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        
        # Subset accuracy
        subset_hit = int(set(y) == set(p))
        subset_acc_hits += subset_hit
        
        # IoU
        inter = tp
        union = len(set(y) | set(p))
        iou = _safe_div(inter, union)
        iou_sum += iou
        
        # Average Precision (if scores provided)
        if pred_scores is not None:
            ap = compute_average_precision(y, pred_scores[idx], num_classes)
            ap_sum += ap
        
        # Per-class counts
        y_set, p_set = set(y), set(p)
        for c in y_set:
            class_has_gt[c] = True
        
        for c in range(num_classes):
            in_y, in_p = (c in y_set), (c in p_set)
            if in_y and in_p:
                class_tp[c] += 1
            elif (not in_y) and in_p:
                class_fp[c] += 1
            elif in_y and (not in_p):
                class_fn[c] += 1
        
        # Bucketed metrics
        bucket = _bucket_of(len(y), bucket_scheme)
        if bucket in buckets:
            buckets[bucket]["tp"] += tp
            buckets[bucket]["fp"] += fp
            buckets[bucket]["fn"] += fn
            buckets[bucket]["subset_hits"] += subset_hit
            buckets[bucket]["count"] += 1
            buckets[bucket]["iou_sum"] += iou
            if pred_scores is not None:
                buckets[bucket]["ap_sum"] += ap
    
    # Compute overall metrics
    micro_prec = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_rec = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec)
    
    subset_acc = subset_acc_hits / N
    mean_iou = iou_sum / N
    mean_ap = ap_sum / N if pred_scores is not None else None
    
    # Macro metrics (over classes with GT)
    per_class_prec, per_class_rec, per_class_f1 = [], [], []
    for c in class_has_gt.keys():
        tp, fp, fn = class_tp[c], class_fp[c], class_fn[c]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        per_class_prec.append(p)
        per_class_rec.append(r)
        per_class_f1.append(f1)
    
    macro_prec = np.mean(per_class_prec) if per_class_prec else 0.0
    macro_rec = np.mean(per_class_rec) if per_class_rec else 0.0
    macro_f1 = np.mean(per_class_f1) if per_class_f1 else 0.0
    
    # Bucketed metrics
    bucket_metrics = {}
    for b, agg in buckets.items():
        cnt = agg["count"]
        if cnt == 0:
            bucket_metrics[b] = {"count": 0}
            continue
        
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        
        bucket_metrics[b] = {
            "count": cnt,
            "subset_acc": agg["subset_hits"] / cnt,
            "mean_iou": agg["iou_sum"] / cnt,
            "micro_precision": p,
            "micro_recall": r,
            "micro_f1": f1,
        }
        
        if pred_scores is not None:
            bucket_metrics[b]["mAP"] = agg["ap_sum"] / cnt
    
    # Build results dictionary
    overall_metrics = {
        "subset_accuracy": subset_acc,
        "mean_iou": mean_iou,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "macro_precision_over_gt_classes": macro_prec,
        "macro_recall_over_gt_classes": macro_rec,
        "macro_f1_over_gt_classes": macro_f1,
    }
    
    if mean_ap is not None:
        overall_metrics["mAP"] = mean_ap
    
    return {
        "n_evaluated_samples": N,
        "overall": overall_metrics,
        "by_gt_label_count": bucket_metrics,
    }


# ============================================================================
# Results Formatting
# ============================================================================

def _get(d: dict, *keys, default=np.nan):
    """Safely navigate nested dictionary."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_multilabel_results(results: Dict[str, dict]) -> pd.DataFrame:
    """Flatten nested results dictionary into a pandas DataFrame."""
    rows = []
    
    for condition, rd in results.items():
        row = {"condition": condition}
        
        # Overall metrics
        row["n_samples"] = _get(rd, "n_evaluated_samples")
        row["overall_subset_acc"] = _get(rd, "overall", "subset_accuracy")
        row["overall_mean_iou"] = _get(rd, "overall", "mean_iou")
        row["overall_mAP"] = _get(rd, "overall", "mAP")
        
        row["overall_micro_precision"] = _get(rd, "overall", "micro_precision")
        row["overall_micro_recall"] = _get(rd, "overall", "micro_recall")
        row["overall_micro_f1"] = _get(rd, "overall", "micro_f1")
        
        row["overall_macro_precision"] = _get(rd, "overall", "macro_precision_over_gt_classes")
        row["overall_macro_recall"] = _get(rd, "overall", "macro_recall_over_gt_classes")
        row["overall_macro_f1"] = _get(rd, "overall", "macro_f1_over_gt_classes")
        
        # Aliases for convenience
        row["global_precision"] = row["overall_micro_precision"]
        row["global_recall"] = row["overall_micro_recall"]
        row["global_f1"] = row["overall_micro_f1"]
        
        # Bucketed metrics - dynamically detect which buckets are present
        # Try both bucket schemes and use whichever is present
        bucket_data = _get(rd, "by_gt_label_count", default={})
        
        # Determine which bucket scheme is used by checking for ">=3" vs "3"
        if isinstance(bucket_data, dict):
            has_simple = ">=3" in bucket_data
            bucket_keys_to_use = BUCKET_KEYS_SIMPLE if has_simple else BUCKET_KEYS_DEFAULT
            
            for bucket_key, prefix in bucket_keys_to_use:
                row[f"{prefix}_count"] = _get(rd, "by_gt_label_count", bucket_key, "count")
                row[f"{prefix}_subset_acc"] = _get(rd, "by_gt_label_count", bucket_key, "subset_acc")
                row[f"{prefix}_mean_iou"] = _get(rd, "by_gt_label_count", bucket_key, "mean_iou")
                row[f"{prefix}_mAP"] = _get(rd, "by_gt_label_count", bucket_key, "mAP")
                row[f"{prefix}_precision"] = _get(rd, "by_gt_label_count", bucket_key, "micro_precision")
                row[f"{prefix}_recall"] = _get(rd, "by_gt_label_count", bucket_key, "micro_recall")
                row[f"{prefix}_f1"] = _get(rd, "by_gt_label_count", bucket_key, "micro_f1")
        
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index("condition").sort_index()
    return df


# ============================================================================
# Prediction Generation
# ============================================================================

def get_predictions_from_logits(
    logits: np.ndarray,
    thresholds: np.ndarray,
    fallback_to_argmax: bool = True
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Convert logits to multi-label predictions using thresholds.
    
    Args:
        logits: (N, C) array of logits
        thresholds: (C,) array of per-class thresholds
        fallback_to_argmax: If True, predict argmax for samples with no positives
    
    Returns:
        predictions: List of predicted label lists
        probs: (N, C) array of probabilities
    """
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds_binary = (probs >= thresholds[None, :]).astype(np.int32)
    predictions = [np.where(row == 1)[0].tolist() for row in preds_binary]
    
    # Fallback to argmax for empty predictions
    if fallback_to_argmax:
        for i, pred_list in enumerate(predictions):
            if len(pred_list) == 0:
                predictions[i] = [int(np.argmax(probs[i]))]
    
    return predictions, probs


# ============================================================================
# Inference
# ============================================================================

def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda:0'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on a dataset.
    
    Returns:
        predictions: (N, C) array of logits
        indices: (N,) array of sample indices
        labels: (N,) array of original labels
    """
    predictions = []
    indices = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            logits = model(batch['image'].to(device))
            predictions.append(logits.cpu().numpy())
            indices.append(batch['index'].cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    indices = np.concatenate(indices, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Sort by index for consistency
    sort_idx = np.argsort(indices)
    predictions = predictions[sort_idx]
    labels = labels[sort_idx]
    indices = indices[sort_idx]
    
    return predictions, indices, labels


# ============================================================================
# Evaluation Routines
# ============================================================================

def evaluate_top1_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    multilabels: Optional[List[List[int]]] = None
) -> Dict[str, float]:
    """
    Compute top-1 accuracy metrics.
    
    Args:
        predictions: (N, C) logits
        labels: (N,) original labels
        multilabels: Optional list of multi-label ground truths
    
    Returns:
        Dictionary of top-1 accuracy metrics
    """
    top1_pred = np.argmax(predictions, axis=1)
    results = {}
    
    # Standard top-1 accuracy
    results['top1_acc'] = float(np.mean(top1_pred == labels))
    
    # Multi-label top-1 accuracy (if provided)
    if multilabels is not None:
        correct = 0
        count = 0
        for pred, gt_list in zip(top1_pred, multilabels):
            if len(gt_list) > 0:
                count += 1
                if pred in gt_list:
                    correct += 1
        if count > 0:
            results['multilabel_top1_acc'] = correct / count
    
    return results


def evaluate_multilabel_with_thresholds(
    predictions: np.ndarray,
    gt_multilabels: List[List[int]],
    threshold_configs: Dict[str, Any],
    num_classes: int = 1000,
    bucket_scheme: str = 'default'
) -> Dict[str, dict]:
    """
    Evaluate multi-label metrics with different threshold configurations.
    
    Args:
        predictions: (N, C) logits
        gt_multilabels: List of ground truth label lists
        threshold_configs: Dict mapping config name to threshold parameters
        num_classes: Number of classes
        bucket_scheme: Either 'default' (1,2,3,>=4) or 'simple' (1,2,>=3)
    
    Returns:
        Dictionary mapping config name to metrics
    """
    results = {}
    
    for config_name, config in threshold_configs.items():
        if config['type'] == 'fixed':
            thresholds = np.full(num_classes, config['value'])
        elif config['type'] == 'calibrated':
            thresholds = calibrate_thresholds_by_count(
                predictions[config.get('indices', slice(None))],
                config['counts']
            )
        else:
            raise ValueError(f"Unknown threshold type: {config['type']}")
        
        preds, probs = get_predictions_from_logits(predictions, thresholds)
        metrics = compute_multilabel_metrics(
            gts=gt_multilabels,
            preds=preds,
            pred_scores=probs,
            num_classes=num_classes,
            bucket_scheme=bucket_scheme
        )
        results[config_name] = metrics
    
    return results


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate_imagenet_val(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: str = 'cuda:0',
    transform: Optional[transforms.Compose] = None
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """Evaluate on ImageNet validation set."""
    print("\n" + "="*80)
    print("Evaluating on ImageNet Validation Set")
    print("="*80)
    
    # Load ground truth labels
    with open(args.real_label_path, 'r') as f:
        real_labels = json.load(f)
    
    with open(args.imagenet_segmentation_label_path, 'r') as f:
        seg_labels = json.load(f)
    
    # Prepare dataset
    if transform is None:
        transform = imagenet_transform(
            size=args.size,
            split='val',
            center_crop=not args.no_center_crop
        )
    dataset = ReaLValSet(root=args.imagenet_val_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16
    )
    
    # Run inference
    logits, indices, ori_labels = run_inference(model, dataloader, device)
    
    # Top-1 evaluation
    top1_results = {}
    top1_metrics = evaluate_top1_accuracy(logits, ori_labels)
    top1_results[f"{args.prefix}_IN1k_val_top1_acc"] = top1_metrics['top1_acc']
    print(f"  Top-1 Accuracy: {top1_metrics['top1_acc']:.4f}")
    
    # ReaL labels top-1
    real_top1 = evaluate_top1_accuracy(logits, ori_labels, real_labels)
    if 'multilabel_top1_acc' in real_top1:
        top1_results[f"{args.prefix}_IN1k_ReaL_val_top1_acc"] = real_top1['multilabel_top1_acc']
        print(f"  ReaL Top-1 Accuracy: {real_top1['multilabel_top1_acc']:.4f}")
    
    # Segmentation labels top-1
    seg_top1 = evaluate_top1_accuracy(logits, ori_labels, seg_labels)
    if 'multilabel_top1_acc' in seg_top1:
        top1_results[f"{args.prefix}_IN1k_INseg_val_top1_acc"] = seg_top1['multilabel_top1_acc']
        print(f"  INseg Top-1 Accuracy: {seg_top1['multilabel_top1_acc']:.4f}")
    
    # Multi-label evaluation with different thresholds
    multilabel_results = {}
    
    # Prepare threshold configurations
    ori_counts = Counter(ori_labels)
    
    # ReaL labels evaluation
    print("\n  Evaluating ReaL multi-label metrics...")
    real_counts = Counter()
    for labels in real_labels:
        for lbl in labels:
            if lbl is not None:  # Handle potential None values
                real_counts[lbl] += 1
    
    _, _, real_filter_idx = _filter_nonempty_gt(real_labels, [[] for _ in real_labels])
    
    real_threshold_configs = {
        f'{args.prefix}_IN1kReaL_thresh@original': {
            'type': 'calibrated',
            'counts': ori_counts
        },
        f'{args.prefix}_IN1kReaL_thresh@val': {
            'type': 'calibrated',
            'counts': real_counts,
            'indices': real_filter_idx
        },
        f'{args.prefix}_IN1kReaL_thresh@0.5': {
            'type': 'fixed',
            'value': 0.5
        }
    }
    
    multilabel_results.update(
        evaluate_multilabel_with_thresholds(
            logits, real_labels, real_threshold_configs,
            bucket_scheme=args.bucket_scheme
        )
    )
    
    # Segmentation labels evaluation
    print("  Evaluating INseg multi-label metrics...")
    seg_counts = Counter()
    for labels in seg_labels:
        for lbl in labels:
            if lbl is not None:
                seg_counts[lbl] += 1
    
    _, _, seg_filter_idx = _filter_nonempty_gt(seg_labels, [[] for _ in seg_labels])
    
    seg_threshold_configs = {
        f'{args.prefix}_IN1kSeg_thresh@original': {
            'type': 'calibrated',
            'counts': ori_counts
        },
        f'{args.prefix}_IN1kSeg_thresh@val': {
            'type': 'calibrated',
            'counts': seg_counts,
            'indices': seg_filter_idx
        },
        f'{args.prefix}_IN1kSeg_thresh@0.5': {
            'type': 'fixed',
            'value': 0.5
        }
    }
    
    multilabel_results.update(
        evaluate_multilabel_with_thresholds(
            logits, seg_labels, seg_threshold_configs,
            bucket_scheme=args.bucket_scheme
        )
    )
    
    return multilabel_results, top1_results


def evaluate_imagenet_v2(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: str = 'cuda:0',
    transform: Optional[transforms.Compose] = None
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """Evaluate on ImageNet-V2 dataset."""
    print("\n" + "="*80)
    print("Evaluating on ImageNet-V2")
    print("="*80)
    
    # Prepare dataset
    if transform is None:
        transform = imagenet_transform(
            size=args.size,
            split='val',
            center_crop=not args.no_center_crop
        )
    dataset = ImageNetV2Dataset(
        root=args.imagenetv2_path,
        annotation_file_path=args.imagenetv2_multilabel_path,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16
    )
    
    # Run inference
    logits, indices, ori_labels = run_inference(model, dataloader, device)
    multilabels = dataset.multilabel
    
    # Top-1 evaluation
    top1_results = {}
    top1_metrics = evaluate_top1_accuracy(logits, ori_labels)
    top1_results[f'{args.prefix}_INv2_val_top1_acc'] = top1_metrics['top1_acc']
    print(f"  Top-1 Accuracy: {top1_metrics['top1_acc']:.4f}")
    
    # Multi-label top-1
    v2_top1 = evaluate_top1_accuracy(logits, ori_labels, multilabels)
    if 'multilabel_top1_acc' in v2_top1:
        top1_results[f'{args.prefix}_INv2_multilabel_top1_acc'] = v2_top1['multilabel_top1_acc']
        print(f"  Multi-label Top-1 Accuracy: {v2_top1['multilabel_top1_acc']:.4f}")
    
    # Multi-label evaluation
    print("  Evaluating multi-label metrics...")
    ori_counts = Counter(ori_labels)
    
    ml_counts = Counter()
    for labels in multilabels:
        for lbl in labels:
            if lbl is not None:
                ml_counts[lbl] += 1
    
    _, _, ml_filter_idx = _filter_nonempty_gt(multilabels, [[] for _ in multilabels])
    
    threshold_configs = {
        f'{args.prefix}_INv2Mul_thresh@original': {
            'type': 'calibrated',
            'counts': ori_counts
        },
        f'{args.prefix}_INv2Mul_thresh@val': {
            'type': 'calibrated',
            'counts': ml_counts,
            'indices': ml_filter_idx
        },
        f'{args.prefix}_INv2Mul_thresh@0.5': {
            'type': 'fixed',
            'value': 0.5
        }
    }
    
    multilabel_results = evaluate_multilabel_with_thresholds(
        logits, multilabels, threshold_configs,
        bucket_scheme=args.bucket_scheme
    )
    
    return multilabel_results, top1_results


# ============================================================================
# Argument Parsing
# ============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate multi-label ImageNet classification'
    )
    
    # General settings
    parser.add_argument(
        '--prefix',
        type=str,
        default='ours',
        help='Prefix for result keys'
    )
    parser.add_argument(
        '--no_center_crop',
        action='store_true',
        help='Do not use center crop for validation'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=224,
        help='Input image size'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--drop',
        type=float,
        default=0.0,
        help='Dropout rate for ViT models'
    )
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.0,
        help='Drop path rate for ViT models'
    )
    # Paths
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--real_label_path',
        type=str,
        default='<DATA_ROOT>/dataset/imagenet/real.json',
        help='Path to ReaL labels JSON'
    )
    parser.add_argument(
        '--imagenet_val_path',
        type=str,
        default='<DATA_ROOT>/dataset/imagenet/val/',
        help='Path to ImageNet validation set'
    )
    parser.add_argument(
        '--imagenet_segmentation_label_path',
        type=str,
        default='<DATA_ROOT>/dataset/imagenet/segmentation/ImageNetS919/real_style_label.json',
        help='Path to ImageNet segmentation labels'
    )
    parser.add_argument(
        '--imagenetv2_path',
        type=str,
        default='<DATA_ROOT>/dataset/imagenet/v2/match_freq/',
        help='Path to ImageNet-V2 dataset'
    )
    parser.add_argument(
        '--imagenetv2_multilabel_path',
        type=str,
        default='<DATA_ROOT>/dataset/imagenet/v2/refined_labels.json',
        help='Path to ImageNet-V2 multi-label annotations'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./evaluation_results/',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet50',
        help='Model architecture',
        # choices=['resnet50', 'resnet101', 'vit_small', 'vit_base', 'vit_large']
    )
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Use finetuned deit3 ViT model if architecture is ViT; used for distinguishing from end2end trained models and finetuned models'
    )
    parser.add_argument(
        '--attach_head',
        action='store_true',
        help='Attach new classification head if output dimension does not match 1000'
    )
    parser.add_argument(
        '--prebuid_transforms',
        action='store_true',
        help='Use prebuilt transforms for evaluation'
    )
    parser.add_argument(
        '--bucket_scheme',
        type=str,
        default='default',
        choices=['default', 'simple'],
        help='Bucket scheme for stratified evaluation: "default" (1,2,3,>=4) or "simple" (1,2,>=3)'
    )
    parser.add_argument(
        '--use_huggingface',
        action='store_true',
        help='Use HuggingFace models for model loading'
    )
    
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main evaluation pipeline."""
    args = get_args()
    
    # Setup device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, trf = load_model(args.checkpoint_path, device, args)
    if not args.prebuid_transforms:
        trf = None # default transforms
    print("Model loaded successfully!")
    
    # Run evaluations
    all_multilabel_results = {}
    all_top1_results = {}
    
    # ImageNet validation
    in_ml_results, in_top1_results = evaluate_imagenet_val(model, args, device, trf)
    all_multilabel_results.update(in_ml_results)
    all_top1_results.update(in_top1_results)
    
    # ImageNet-V2
    v2_ml_results, v2_top1_results = evaluate_imagenet_v2(model, args, device, trf)
    all_multilabel_results.update(v2_ml_results)
    all_top1_results.update(v2_top1_results)
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    if args.checkpoint_path is not None:
        checkpoint_name = os.path.basename(args.checkpoint_path).replace('.pth', '').replace('/', '_')
    else:
        checkpoint_name = 'official_model'
    save_dir = os.path.join(args.save_path, f'{args.prefix}_{checkpoint_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save multi-label metrics as CSV
    df = flatten_multilabel_results(all_multilabel_results)
    csv_path = os.path.join(save_dir, 'multilabel_metrics.csv')
    df.to_csv(csv_path)
    print(f"  Saved multi-label metrics to: {csv_path}")
    
    # Save top-1 metrics as JSON
    json_path = os.path.join(save_dir, 'top1_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_top1_results, f, indent=2)
    print(f"  Saved top-1 metrics to: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    print("\nTop-1 Accuracies:")
    for key, value in all_top1_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMulti-label Metrics (Overall mAP):")
    for condition in all_multilabel_results.keys():
        map_score = _get(all_multilabel_results[condition], "overall", "mAP")
        if not np.isnan(map_score):
            print(f"  {condition}: {map_score:.4f}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()

