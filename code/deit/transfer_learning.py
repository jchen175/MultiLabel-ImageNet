import os
import torch.distributed as dist
import wandb
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from torchvision import models
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from util import COCOMultiLabelDataset, evaluate_multilabel_ddp, EvalConfig, load_cached_index, load_state_dict_with_ddp_handling
from timm import create_model
from timm.models import create_model as models_create_model
# handle deitv3 models; git clone https://github.com/facebookresearch/deit.git
import models as models_v1
import models_v2
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

condition_ckpt_mapping = {
    # add more mappings as needed: 'condition_name': [ckpt_path, arch, finetune_from_official_ckpt(bool), pretrain_size(int)]
    'e2e_vitb_384': ['./checkpoints/e2e_finetune_20ep_vitb384/checkpoint.pth', 'deit_base_patch16_LS', False, 384],
}



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


# ---------- original control flow, adapted for huggingface model----------


def build_model(arch, num_classes: int, attach_head: bool = False) -> nn.Module:
    """
    args.arch: Hugging Face model id (e.g., 'google/vit-base-patch16-224')
    args.attach_head: bool
    """
    model_id = arch
    cfg = AutoConfig.from_pretrained(model_id)

    if attach_head:
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



_BACKBONE_HEAD_PREFIXES = (
    "backbone.fc",          # torchvision ResNet
    "backbone.head",        # many timm models (e.g., ViT)
    "backbone.classifier",  # some MobileNet/EfficientNet variants
    "backbone.cls_head",    # some timm models
    "backbone.last_linear", # some older repos
)

def _is_backbone_head_param(name: str) -> bool:
    return any(name.startswith(pref) for pref in _BACKBONE_HEAD_PREFIXES)

def build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    head_lr_mul: float = 1.0,
    nodecay_head: bool = True,
    nodecay_norm_bias: bool = False,
):
    # Collect params
    new_head_params = list(model.new_head.parameters())

    backbone_head_params = [p for n, p in model.named_parameters() if _is_backbone_head_param(n)]

    # Optionally split out norms/biases to no-decay as well (common trick)
    def is_norm_or_bias(mod_name, mod, pname):
        if not nodecay_norm_bias:
            return False
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn import LayerNorm, GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
        norm_types = (_BatchNorm, LayerNorm, GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d)
        return isinstance(mod, norm_types) or pname.endswith("bias")

    # Map: param -> group tag
    head_set = {id(p) for p in new_head_params} | {id(p) for p in backbone_head_params}

    decay_params, no_decay_params = [], []
    head_decay_params, head_no_decay_params = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # find module for norm/bias check
        mod = model
        for part in n.split(".")[:-1]:
            mod = getattr(mod, part)
        pname = n.split(".")[-1]

        if id(p) in head_set:
            # all "head" params (original + new) go here
            if nodecay_head or is_norm_or_bias(n, mod, pname):
                head_no_decay_params.append(p)
            else:
                head_decay_params.append(p)
        else:
            # backbone non-head
            if is_norm_or_bias(n, mod, pname):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": base_lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0})
    if head_decay_params:
        groups.append({"params": head_decay_params, "lr": base_lr * head_lr_mul, "weight_decay": weight_decay})
    if head_no_decay_params:
        groups.append({"params": head_no_decay_params, "lr": base_lr * head_lr_mul, "weight_decay": 0.0})

    return groups


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
    out = torch.full_like(y, neg_val, device=y.device, dtype=y.dtype)

    # for pos locations: cap at pos_val
    pos_mask = y > neg_val
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

def get_train_val_transforms(size=224, scale_min = 0.08, val_center_crop=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if val_center_crop:
        val_transform_list = [
            transforms.Resize((int(size**1.15), int(size*1.15))),
            transforms.CenterCrop(size)
        ]
    else:
        val_transform_list = [
            transforms.Resize((size, size))
        ]
    val_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose(val_transform_list)
    return train_transform, val_transform


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dataset_root", type=str, default="<DATA_ROOT>/dataset/mscoco/")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--save_dir", type=str, default="./ckpt/transfer_multilabel")
    parser.add_argument('--condition', type=str, choices=list(condition_ckpt_mapping.keys()), help='condition ckpt mapping')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/', help='path to the pretrained checkpoint')
    parser.add_argument('--debug', action='store_true')

    # other args
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--head_lr_mul', type=float, default=1.0, help='lr multiplier for classification head')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "reduce_on_plateau", 'cosine'])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument('--patience', type=int, default=8, help='reduce_on_plateau patience')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR step size (epochs)')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR/reduce_op gamma')
    parser.add_argument('--use_pos_weight', type=str, default='false',
                        help='whether to use pos_weight in BCEWithLogitsLoss; call str2bool as a workaround for sweep')
    parser.add_argument('--label_smoothing_min', type=float, default=0.0, help='label smoothing for negative class')
    parser.add_argument('--label_smoothing_max', type=float, default=1.0, help='label smoothing for positive class')
    parser.add_argument('--nodecay', type=str, default='false',
                        help='whether to ignore weight decay on the classification head; call str2bool as a workaround for sweep')
    # parser.add_argument('--attach_linear', type=str, default='false', help='whether to attach a linear layer after the classification head, instead of replacing it')
    parser.add_argument('--project', type=str, default='multilabel-classification-transfer', help='wandb project name')
    parser.add_argument('--name', type=str, default=None, help='name of experiment')
    parser.add_argument('--group', type=str, default=None, help='group name for wandb')

    # image size and scale
    parser.add_argument('--size', type=int, default=224, help='input image size')
    parser.add_argument('--scale_min', type=float, default=0.08, help='min scale for RandomResizedCrop')
    parser.add_argument('--val_center_crop', type=str, default='false', help='whether to use center crop for val')
    # drop
    parser.add_argument('--drop', type=float, default=0.1, help='drop rate')
    # drop_path
    parser.add_argument('--drop_path', type=float, default=0.1, help='drop path rate')

    # alternative loss
    parser.add_argument('--loss', type=str, choices=['asl', 'bce', 'ral'], default='bce')
    parser.add_argument('--gamma_neg', type=float, default=4, help='used only if --loss asl/ral')
    parser.add_argument('--gamma_pos', type=float, default=1, help='used only if --loss asl/ral')
    parser.add_argument('--clip', type=float, default=0.05, help='used only if --loss asl/ral')
    parser.add_argument('--resume_from', type=str, default=None, help='path to checkpoint to resume from')
    return parser.parse_args()

def get_exp_name(args):
    if args.loss == 'bce':
        loss_prefix = f"BCE-LSM@{args.label_smoothing_min}-{args.label_smoothing_max}-PW@{args.use_pos_weight}"
    elif args.loss == 'asl' or args.loss=='ral':
        loss_prefix = f"{args.loss.upper()}@gn{args.gamma_neg}-gp{args.gamma_pos}-c{args.clip}"
        if args.label_smoothing_min != 0 or args.label_smoothing_max != 1.0:
            print("Warning: label smoothing is not typically used with ASL loss.")
            print("Overwriting with 0 and 1.0")
            args.label_smoothing_min = 0.0
            args.label_smoothing_max = 1.0
    if 'coco' in args.dataset_root.lower():
        dataset_name = 'coco'
        if not os.path.exists(args.dataset_root):
            # retrieve from environment variable
            if 'COCO_DATASET' in os.environ:
                args.dataset_root = os.environ['COCO_DATASET']
            else:
                raise ValueError("COCO dataset path does not exist and COCO_DATASET env variable not set.")

    elif 'voc' in args.dataset_root.lower():
        dataset_name = 'voc'
        if not os.path.exists(args.dataset_root):
            # retrieve from environment variable
            if 'VOC_DATASET' in os.environ:
                args.dataset_root = os.environ['VOC_DATASET']
            else:
                raise ValueError("VOC dataset path does not exist and VOC_DATASET env variable not set.")
    else:
        dataset_name = 'custom'
    args.dataset_name = dataset_name
    name = (f"{dataset_name}_ckpt@{args.condition}_{loss_prefix}_scaleMin@{args.scale_min}_"
            f"lr@{args.lr}_headMul@{args.head_lr_mul}_AddLH@True_Gbs@{args.global_batch_size}_"
            f"op@{args.optimizer}WD@{args.weight_decay}NoDHead@{args.nodecay}_"
            f"sche@{args.scheduler}_ep@{args.num_epochs}")
    return name.replace('/', '_').replace(' ', '')


def main():
    # Initialize DDP
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    args = get_args()
    args.global_batch_size = args.batch_size
    args.batch_size = args.global_batch_size // world_size
    args.nodecay = str2bool(args.nodecay)
    args.use_pos_weight = str2bool(args.use_pos_weight)
    args.val_center_crop = str2bool(args.val_center_crop)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.name is None:
        args.name = get_exp_name(args)

    # Initialize wandb on rank 0 for experiment tracking
    if rank == 0 and not args.debug:
        wandb.init(project=args.project, entity='<YOUR_WANDB_ENTITY>', name=args.name, group=args.group)
        wandb.config.update(args)
    # get dataset
    train_samples, train_meta = load_cached_index(os.path.join(args.dataset_root, 'multilabel', 'train.json'))
    val_samples, val_meta = load_cached_index(os.path.join(args.dataset_root, 'multilabel', 'val.json'))
    train_transform, val_transform = get_train_val_transforms(size=args.size, scale_min=args.scale_min, val_center_crop=args.val_center_crop)
    train_dataset = COCOMultiLabelDataset(train_samples, train_meta, transform=train_transform)
    val_dataset = COCOMultiLabelDataset(val_samples, val_meta, transform=val_transform)

    num_classes = train_meta['num_classes']

    # Distributed samplers for multi-GPU training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # DataLoaders (note: pin_memory=True for faster host-to-GPU transfers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)


    # prepare model for transfer learning
    out_dim = 1000
    ckpt_path, arch, finetune, pretrain_size = condition_ckpt_mapping[args.condition]

    if arch == 'resnet50':
        model = models.resnet50(weights=None if not finetune else models.ResNet50_Weights.IMAGENET1K_V2,
                                num_classes=1000)
    elif arch == 'resnet101':
        model = models.resnet101(weights=None if not finetune else models.ResNet101_Weights.IMAGENET1K_V2,
                                 num_classes=1000)
    elif 'deit' in arch.lower() and not arch.startswith('timm/'):
        model = models_create_model(
            arch,
            pretrained=False,
            num_classes=1000,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.size
        )

    elif 'vit' in arch and finetune:
        model = create_model(f"timm/deit3_{arch.split('_')[1]}_patch16_{pretrain_size}.fb_in1k",
                             pretrained=True)
    elif 'google/' in arch or 'hf/' in arch:
        model = build_model(arch, num_classes=1000, attach_head='_ahTrue' in args.condition)
    else: # timm models
        model = create_model(arch, pretrained=True)
        out_dim = model.get_classifier().out_features
        if out_dim != 1000 and ckpt_path is not None:
            print(f"num_classes of loaded model is {out_dim}, modifying to 1000.")
            if 'AH@True' in ckpt_path:
                model = nn.Sequential(OrderedDict([
                    ("backbone", model),
                    ("new_head", nn.Linear(out_dim, 1000)),
                ]))
            else:
                # modify the existing head
                model.reset_classifier(num_classes=1000)
            out_dim = 1000

    if ckpt_path is not None:
        loaded = False
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, weights_only=False)
        else:
            checkpoint = torch.load(os.path.join(args.ckpt_root, ckpt_path), weights_only=False)
        for key in ['model', 'model_dict', 'model_state_dict']:
            if key in checkpoint:
                load_state_dict_with_ddp_handling(model, checkpoint[key], verbose=(local_rank == 0))
                loaded = True
                break
        if not loaded:
            # If no standard key found, try loading directly
            load_state_dict_with_ddp_handling(model, checkpoint, verbose=(local_rank == 0))

    # if args.attach_linear:
    model = nn.Sequential(OrderedDict([
        ("backbone", model),
        ("new_head", nn.Linear(out_dim, num_classes)),
    ]))
    model.to(device)


    # Load checkpoint if resuming
    start_epoch = 0

    checkpoint_loaded = False
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from}")

        if local_rank == 0:
            print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

        # Load model states with DDP handling
        load_state_dict_with_ddp_handling(model, checkpoint['model_state_dict'],
                                          verbose=(local_rank == 0))
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        checkpoint_loaded = True

        if local_rank == 0:
            print(f"Successfully loaded classification head state dict")
            print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")

    # Wrap model for distributed training
    if args.loss == 'bce':
        # Loss function for multi-label (Binary Cross-Entropy with logits)
        if args.use_pos_weight:
            Y = torch.stack([s["target"] for s in train_samples], 0)  # [N, 20]
            pos, neg = Y.sum(0), Y.shape[0] - Y.sum(0)
            eps = 1e-6
            pos_weight = ((neg + eps) / (pos + eps)).clamp(max=50.0)
            pos_weight = pos_weight.to(device)
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == 'asl':
        criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip)
    elif args.loss == 'ral':
        criterion = Ralloss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip)

    else:
        raise ValueError(f"Unknown loss {args.loss}")

    param_groups = build_param_groups(
        model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        head_lr_mul=getattr(args, "head_lr_mul", 1.0),
        nodecay_head=args.nodecay,  # <— no decay for BOTH original & new heads
        nodecay_norm_bias=getattr(args, "nodecay_norm_bias", False)  # optional
    )

    opt_name = args.optimizer.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # "step", "reduce_on_plateau", 'cosine'
    if args.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, patience=args.patience)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=0)


    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    if checkpoint_loaded:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    best_mAP = 0.0
    best_f1_micro = 0.0
    best_f1_macro = 0.0
    for epoch in range(start_epoch, args.num_epochs):
        is_last_epoch = (epoch == args.num_epochs - 1)
        # Learning rate scheduling
        if epoch < args.warmup_epochs:
            # Linear LR warmup phase
            warmup_factor = float(epoch + 1) / float(args.warmup_epochs)
            new_lr = args.lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            if rank == 0:
                print(f"Warmup epoch {epoch + 1}: learning rate set to {new_lr:.6f}")
        else:
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(results['f1_micro'])
            else:
                scheduler.step()

        # ---- Training Phase ----
        model.train()
        train_sampler.set_epoch(epoch)  # ensure new shuffle for this epoch (DDP requirement)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} - Training...")
            iterator = tqdm(train_loader)
        else:
            iterator = train_loader

        for batch_step, (x, y, _) in enumerate(iterator):
            images = x.to(device)
            targets = y.to(device)
            with autocast():
                logits = model(images)
                if args.label_smoothing_min > 0.0 or args.label_smoothing_max < 1.0:
                    targets = smooth_targets_cap(targets, pos_val=args.label_smoothing_max, neg_val=args.label_smoothing_min)
                loss = criterion(logits, targets)

            # Backprop + step with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                iterator.set_description(f"Train Loss: {loss.item():.4f}")
                wandb.log({
                    "Train/Train Loss": float(loss.item()),
                    "Train/Learning Rate": optimizer.param_groups[0]['lr'],
                    "Train/GradScaler Scale": scaler.get_scale(),
                })

            if args.debug and batch_step >= 10:
                break
        dist.barrier()  # ensure all processes have finished training phase

        # ---- Validation Phase ----
        model.eval()
        if is_last_epoch:
            eva_cfg = EvalConfig(
                threshold=0.5,
                sweep_best_micro_f1=True,
                sweep_grid=(0.05, 0.95, 0.05),
                compute_auc=False,
            )
        else:
            eva_cfg = EvalConfig(threshold=0.5, sweep_best_micro_f1=False, compute_auc=False)

        dist.barrier()  # ensure all processes have finished validation
        results = evaluate_multilabel_ddp(model, val_loader, num_classes, eva_cfg, device=device)
        best_mAP = max(best_mAP, results['mAP'])
        best_f1_micro = max(best_f1_micro, results['f1_micro'])
        best_f1_macro = max(best_f1_macro, results['f1_macro'])
        # ---- Logging (only on rank 0) ----
        if rank == 0:
            logged_results = {
                "val/epoch": epoch,
                "val/precision_micro": results['precision_micro'],
                "val/recall_micro": results['recall_micro'],
                "val/f1_micro": results['f1_micro'],
                "val/precision_macro": results['precision_macro'],
                "val/recall_macro": results['recall_macro'],
                "val/f1_macro": results['f1_macro'],
                "best/best_mAP": best_mAP,
                "best/best_f1_micro": best_f1_micro,
                "best/best_f1_macro": best_f1_macro,
            }
            wandb.log(logged_results)
            # save checkpoint
            checkpoint_path = f"{args.name}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'eval_results': results,
                'args': args,
            }, os.path.join(args.save_dir, checkpoint_path))
        dist.barrier()
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()