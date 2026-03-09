from PIL import Image
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import torch.distributed as dist

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Multi-label COCO dataset and utils
def build_coco_multilabel_index(
    ann_file: str,
    img_root: str,
    cache_path: str = None,
    drop_images_without_labels: bool = True,
):
    """
    Returns:
      samples: list of dicts:
        {
          "image_id": int,
          "path": str,
          "target": torch.FloatTensor [80],  # multi-hot
        }
      meta: dict with 'cats': [{'id': coco_cat_id, 'name': name, 'idx': 0..79}],
            and id->idx map
    """
    from pycocotools.coco import COCO
    coco = COCO(ann_file)

    # 80 "thing" classes used for detection (skip "stuff"). COCO cat ids are not contiguous; remap to [0..79].
    cat_ids = sorted(coco.getCatIds())                 # e.g., [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, ...]
    cats = coco.loadCats(cat_ids)
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    cats_sorted = [{"id": c["id"], "name": c["name"], "idx": id2idx[c["id"]]} for c in cats]

    # Collect image -> set(category_idx)
    img_to_cats = defaultdict(set)
    ann_ids = coco.getAnnIds()  # all instance annotations
    anns = coco.loadAnns(ann_ids)
    for a in anns:
        if a.get("iscrowd", 0) == 1:
            continue
        cid = a["category_id"]
        if cid in id2idx:
            img_to_cats[a["image_id"]].add(id2idx[cid])

    # Build samples
    img_root = Path(img_root)
    img_infos = coco.loadImgs(sorted(coco.getImgIds()))
    samples = []
    for info in img_infos:
        path = img_root / info["file_name"]
        label_idxs = sorted(list(img_to_cats.get(info["id"], [])))
        if drop_images_without_labels and not label_idxs:
            continue
        target = torch.zeros(len(cat_ids), dtype=torch.float32)
        if label_idxs:
            target[label_idxs] = 1.0
        samples.append({
            "image_id": int(info["id"]),
            "path": str(path),
            "target": target,
        })

    meta = {"cats": cats_sorted, "id2idx": id2idx, "num_classes": len(cat_ids)}

    if cache_path:
        out = {"samples": [
                    {"image_id": s["image_id"], "path": s["path"], "target": s["target"].tolist()}
                for s in samples
              ],
               "meta": meta}
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)

    return samples, meta


def load_cached_index(cache_path: str):
    with open(cache_path, "r") as f:
        data = json.load(f)
    samples = []
    for s in data["samples"]:
        samples.append({
            "image_id": s["image_id"],
            "path": s["path"],
            "target": torch.tensor(s["target"], dtype=torch.float32)
        })
    return samples, data["meta"]


class COCOMultiLabelDataset(Dataset):
    def __init__(self, samples, meta, transform=None):
        self.samples = samples
        self.meta = meta
        self.tf = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("RGB")
        if self.tf is not None:
            img = self.tf(img)
        y = item["target"]  # FloatTensor [80], multi-hot
        return img, y, {"image_id": item["image_id"], "path": item["path"]}


# Multi-label for VOC dataset and utils
VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow","diningtable",
    "dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]
NAME2IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

def _gather_image_ids(voc_root: str, year: str, split: str):
    # split in {"train","val","trainval","test"}
    f = Path(voc_root)/f"VOC{year}"/"ImageSets"/"Main"/f"{split}.txt"
    ids = [x.strip() for x in open(f).read().splitlines() if x.strip()]
    return ids

def _one_img_labels(xml_path: Path, use_difficult: bool):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labs = set()
    for obj in root.findall("object"):
        diff = int(obj.findtext("difficult","0"))
        if diff == 1 and not use_difficult:
            continue
        name = obj.findtext("name").strip().lower()
        if name in NAME2IDX:
            labs.add(NAME2IDX[name])
    return sorted(labs)

def build_voc_multilabel_index(
    voc_root: str,               # path to VOCdevkit
    year: str = "2007",          # "2007" or "2012"
    split: str = "trainval",     # "train"|"val"|"trainval"|"test"
    use_difficult: bool = False, # include <difficult> objects?
    drop_images_without_labels: bool = True,
    cache_path: str = None,
):
    """
    Returns:
      samples: list of dicts: {"image_id": str, "path": str, "target": FloatTensor[20]}
      meta: {"classes": VOC_CLASSES, "num_classes": 20, "year": year, "split": split}
    """
    root = Path(voc_root)/f"VOC{year}"
    ids = _gather_image_ids(voc_root, year, split)
    samples = []
    for img_id in ids:
        xml = root/"Annotations"/f"{img_id}.xml"
        labels = _one_img_labels(xml, use_difficult)
        if drop_images_without_labels and not labels:
            continue
        target = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        if labels:
            target[labels] = 1.0
        img_path = (root/"JPEGImages"/f"{img_id}.jpg").as_posix()
        samples.append({"image_id": img_id, "path": img_path, "target": target})

    meta = {"classes": VOC_CLASSES, "num_classes": len(VOC_CLASSES),
            "year": year, "split": split, "use_difficult": use_difficult}

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"samples":[{"image_id":s["image_id"],"path":s["path"],
                                   "target":s["target"].tolist()} for s in samples],
                       "meta": meta}, f)
    return samples, meta




# DDP evaluation
def init_distributed():
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # single-process fallback
        rank = 0; world_size = 1; local_rank = 0
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def gather_arrays_across_processes(obj):
    """
    Robust all-gather for arbitrary Python objects (lists/ndarrays/tensors on CPU).
    Requires PyTorch >= 1.10.
    """
    if not dist.is_initialized():
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out

# -------------------- Metric helpers -------------------- #
import numpy as np

def _precision_recall_curve_binary(y_true: np.ndarray, y_score: np.ndarray):
    """
    y_true: (N,) in {0,1}
    y_score: (N,) real-valued scores (higher = more positive)
    Returns precision, recall, thresholds following sklearn's spirit:
      precision: (M+1,), recall: (M+1,), thresholds: (M,)
    """
    # Sort by decreasing score
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]

    # True positives accumulated
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    # avoid divide by zero
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp[-1] if tp.size > 0 else 0, 1)

    # Prepend start point (P=1 at R=0) like sklearn
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    # thresholds size == number of scores (unique or not). We follow sklearn policy:
    thresholds = y_score[order]
    return precision, recall, thresholds

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AP = integral of precision w.r.t recall, using interpolation to make precision non-increasing.
    Equivalent to sklearn.metrics.average_precision_score (no sampling noise handling).
    """
    if y_true.sum() == 0:
        return np.nan  # undefined (no positives); caller can skip in mean
    precision, recall, _ = _precision_recall_curve_binary(y_true, y_score)
    # Make precision non-increasing (interp)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    # Area under PR curve (R as x-axis). Use trapezoidal rule on recall changes.
    # Note: sklearn uses step-wise integration; both are fine. We’ll do step-wise:
    ap = 0.0
    for i in range(1, len(precision)):
        delta_r = recall[i] - recall[i - 1]
        ap += precision[i] * max(delta_r, 0.0)
    return float(ap)

def binarize(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores >= thr).astype(np.int32)

def precision_recall_f1(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, average: str) -> Tuple[float,float,float]:
    """
    y_true_bin, y_pred_bin: shape (N, C) in {0,1}
    average: 'micro' or 'macro'
    """
    eps = 1e-9
    if average == "micro":
        tp = np.logical_and(y_true_bin == 1, y_pred_bin == 1).sum()
        fp = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
        fn = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, eps) if (p + r) > 0 else 0.0
        return float(p), float(r), float(f1)
    elif average == "macro":
        C = y_true_bin.shape[1]
        ps, rs, f1s = [], [], []
        for c in range(C):
            yt, yp = y_true_bin[:, c], y_pred_bin[:, c]
            tp = np.logical_and(yt == 1, yp == 1).sum()
            fp = np.logical_and(yt == 0, yp == 1).sum()
            fn = np.logical_and(yt == 1, yp == 0).sum()
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, eps) if (p + r) > 0 else 0.0
            ps.append(p); rs.append(r); f1s.append(f1)
        return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))
    else:
        raise ValueError("average must be 'micro' or 'macro'.")

def try_sklearn_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    try:
        from sklearn.metrics import roc_auc_score
        per_class_auc = []
        C = y_true.shape[1]
        for c in range(C):
            yt = y_true[:, c]
            if yt.max() == yt.min():  # no pos/neg mix
                per_class_auc.append(np.nan)
            else:
                per_class_auc.append(roc_auc_score(yt, y_score[:, c]))
        macro_auc = np.nanmean(per_class_auc)
        # micro-auc
        if y_true.sum() == 0 or (y_true.shape[0]*y_true.shape[1] - y_true.sum()) == 0:
            micro_auc = float("nan")
        else:
            micro_auc = roc_auc_score(y_true.ravel(), y_score.ravel())
        return np.array(per_class_auc, dtype=float), float(macro_auc), float(micro_auc)
    except Exception:
        return None

# -------------------- Evaluator -------------------- #
@dataclass
class EvalConfig:
    threshold: float = 0.5
    sweep_best_micro_f1: bool = True
    sweep_grid: Tuple[float, float, float] = (0.05, 0.95, 0.05)  # start, end, step
    compute_auc: bool = True  # will try sklearn if available

def evaluate_multilabel_ddp(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    cfg: EvalConfig = EvalConfig(),
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Returns dict with:
      - mAP, per_class_AP
      - micro/macro precision/recall/f1 at cfg.threshold
      - (optional) best_micro_f1 & best_threshold from a sweep
      - (optional) per_class_auc, macro_auc, micro_auc (if sklearn available)
    """
    init_distributed()
    model.eval()
    device = device or torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

    all_scores_local = []
    all_targets_local = []

    with torch.no_grad():
        for batch in loader:
            # Expect (images, targets, meta) or (images, targets)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            else:
                raise ValueError("Loader must return (images, targets, ...)")
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()

            logits = model(images)
            scores = torch.sigmoid(logits).float()

            all_scores_local.append(scores.detach().cpu())
            all_targets_local.append(targets.detach().cpu())

    # stack locally
    scores_local = torch.cat(all_scores_local, dim=0).numpy() if all_scores_local else np.zeros((0, num_classes), dtype=np.float32)
    targets_local = torch.cat(all_targets_local, dim=0).numpy() if all_targets_local else np.zeros((0, num_classes), dtype=np.float32)

    # gather across processes
    gathered_scores = gather_arrays_across_processes(scores_local)
    gathered_targets = gather_arrays_across_processes(targets_local)

    # concat along first dim
    scores = np.concatenate(gathered_scores, axis=0) if len(gathered_scores) > 0 else scores_local
    targets = np.concatenate(gathered_targets, axis=0) if len(gathered_targets) > 0 else targets_local

    # --- AP / mAP ---
    per_class_AP = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        ap = average_precision(targets[:, c].astype(np.int32), scores[:, c].astype(np.float32))
        per_class_AP[c] = np.nan if ap is None else ap
    mAP = float(np.nanmean(per_class_AP)) if (np.isfinite(per_class_AP).any()) else float("nan")

    # --- PR/F1 at fixed threshold ---
    thr = cfg.threshold
    y_pred_bin = binarize(scores, thr)
    p_micro, r_micro, f1_micro = precision_recall_f1(targets, y_pred_bin, "micro")
    p_macro, r_macro, f1_macro = precision_recall_f1(targets, y_pred_bin, "macro")

    # --- Optional threshold sweep for best micro-F1 ---
    best_f1 = None; best_thr = None
    if cfg.sweep_best_micro_f1:
        start, end, step = cfg.sweep_grid
        t = start
        best_f1 = -1.0; best_thr = start
        while t <= end + 1e-12:
            yp = binarize(scores, t)
            _, _, f1 = precision_recall_f1(targets, yp, "micro")
            if f1 > best_f1:
                best_f1, best_thr = f1, t
            t += step
        best_f1 = float(best_f1); best_thr = float(best_thr)

    # --- Optional ROC-AUC via sklearn ---
    auc_stats = None
    if cfg.compute_auc:
        auc_stats = try_sklearn_auc(targets, scores)  # (per_class_auc, macro_auc, micro_auc) or None

    results = {
        "mAP": mAP,
        "per_class_AP": per_class_AP,   # np.array [C]
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
    }
    if cfg.sweep_best_micro_f1:
        results["best_micro_f1"] = best_f1
        results["best_threshold"] = best_thr
    if auc_stats is not None:
        per_class_auc, macro_auc, micro_auc = auc_stats
        results["per_class_auc"] = per_class_auc
        results["macro_auc"] = macro_auc
        results["micro_auc"] = micro_auc

    if is_main():
        # Pretty print a short summary
        k = "f1_micro"
        print(f"[Eval] N={scores.shape[0]} | C={num_classes} | mAP={mAP:.4f} | micro-F1@{thr:.2f}={results[k]:.4f}")
        if cfg.sweep_best_micro_f1:
            print(f"[Eval] best_micro_F1={best_f1:.4f} at threshold={best_thr:.2f}")

    return results
