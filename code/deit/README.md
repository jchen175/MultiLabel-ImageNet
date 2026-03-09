# DeiT - Modified Implementation

Third-party implementation: https://github.com/facebookresearch/deit

## Modifications

### `main_ddp.py`
Revised from original implementation to run locally without submitting to a SLURM cluster. Mixup/cutmix implementation has been switched from timm to torchvision to support multi-label.

### `datasets.py`
Adds implementation of multi-label ImageNet dataset class with support for loading label mappings from JSON/PT files.

### `evaluation.py`
Evaluates a model checkpoint (works for both ViT and ResNet) on ImageNet and variants. Provides both top-1 accuracy and overall/subgroup mAP metrics for multi-label evaluation.

### `transfer_learning.py`
Transfer a pretrained model on COCO/VOC. Supports both ViT and ResNet architectures.

