# Unlocking ImageNet’s Multi-Object Nature: Automated Large-Scale Multilabel
This directory contains the code implementation for the paper.

The current repository provides a complete implementation of the full pipeline used in our experiments. We are continuing to refine the structure, documentation, and performance.

## Directory Structure

```
code/
├── CutLER/                  # Modified CutLER for object proposal generation
│   ├── maskcut/
│   │   ├── maskcut_dinov3.py       # Add DINOv2/v3 support
│   │   └── merge_scattered_jsons.py # Merge split job results
│   └── README.md
│
├── deit/                    # Modified DeiT for multi-label training/evaluating
│   ├── datasets.py                 # Add Multi-label dataset classes
│   ├── evaluation.py               # Evaluation metrics for multi-label ImageNet
│   ├── main_ddp.py                 # Local DDP training script
│   ├── transfer_learning.py        # Transfer to COCO/VOC
│   └── README.md
│
├── labeler/                 # Classification head training
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── base_datasets.py          # Dataset utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_hook.py           # Model wrappers for feature extraction
│   │   ├── pretrained_encoder.py   # Load pre-trained encoders
│   │   └── cls_head.py        # Classification head architectures
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── base_trainer.py         # Training utilities
│   ├── mask_level_relabel.py       # Derive confidence scores from ReLabel
│   ├── train_labeler.py            # Train classification head
│   ├── relabel_trainset.py         # Inference on proposals
│   └── README.md
│
├── resnet_exp/              # ResNet experiments
│   ├── train_w_relabel.py          # Train with multi-label
│   └── README.md
│
├── env/                     # Environment Setup
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
│
└── README.md               # This file
```

## Key Components

### 1. Object Proposal Generation (CutLER/)

Generate pseudo-masks using MaskCut algorithm with DINO features:

- `maskcut_dinov3.py`: Add implementation for DINOv2/v3 models
- Output format: COCO-style JSON with RLE-encoded masks

### 2. Classification Head Training (labeler/)

Train a classification head on masked regions:
1. `mask_level_relabel.py`: Filter proposals based on confidence from ReLabel 
2. `train_labeler.py`: Train head on frozen backbone with filtered proposals
3. `relabel_trainset.py`: Apply trained head to all proposals

### 3. Multi-Label Training (deit/)

Train vision transformers with multi-label supervision:
- `main_ddp.py`: Distributed training with multi-label support
- `datasets.py`: Add multi-label ImageNet dataset loader
- `evaluation.py`: Comprehensive evaluation metrics with subgroup analysis

### 4. ResNet Baseline (resnet_exp/)

ResNet experiments comparing single-label vs multi-label training:
- `train_w_relabel.py`: DDP training script
- For evaluation and transfer learning, use `../deit/evaluation.py`, and `../deit/transfer_learning.py` scripts.

## Typical Workflow Example

1. **Generate Object Proposals**
   
   ```bash
   python CutLER/maskcut/maskcut_dinov3.py --dataset_path <DATA_ROOT>/imagenet/ \
   								 		--split train
   ```
   
2. **Derive Mask Confidence**
   
   ```bash
   python labeler/mask_level_relabel.py --segmentation_file proposals.json \
                                        --relabel_root <DATA_ROOT>/imagenet/ReLabel
   ```
   
3. **Train Classification Head**
   
   ```bash
   torchrun --nproc_per_node=4 labeler/train_labeler.py \
            --seg_annotation proposals.json \
            --mask_label_dict confidence_scores.pt
   ```
   
4. **Derive Multi-Label Training Set**
   
   ```bash
   python labeler/relabel_trainset.py \
            --checkpoint_path trained_head.pth \
            --annotation_file proposals.json
   ```
   
5. **Multi-Label Training**
   ```bash
   torchrun --nproc_per_node=8 deit/main_ddp.py \
            --data-path <DATA_ROOT>/imagenet \
            --data-set MULTILABEL \
            --relabel_json relabel_results.json
   ```
   
6. **Evaluation**
   ```bash
   python deit/evaluation.py --checkpoint best_model.pth \
                             --imagenet_val_path <DATA_ROOT>/imagenet/val \
   						  --imagenetv2_path <DATA_ROOT>/imagenetv2/match_freq
   ```
6. **Transfer Learning**
   
   ```bash
   python deit/transfer_learning.py --dataset_root <DATA_ROOT>/mscoco \
                             --condition e2e_multilabel \
   						  --ckpt_root ./ckpt/
   ```


## Dependencies

See `./env`.

## Notes

- All training scripts support Distributed Data Parallel with mixed precision training.
- For multi-GPU training, use `torchrun` or `torch.distributed.launch`
