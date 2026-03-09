import os
import argparse
import time
import json
import torch
from torch import nn
import wandb
from torch.utils.data import DataLoader
from datasets.base_datasets import get_mask_augmentation, RelabelMaskcutDataset, imagenet_transform_cpu, ReaLValSet, ImageNetHelper
from models.cls_head import get_cls_head
from models.pretrained_encoder import get_dinov3_models
from trainers.base_trainer import get_optimizer, get_stepwise_scheduler, seed_everything, EarlyStoppingDDP
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from util import load_state_dict_with_ddp_handling, get_masked_prediction

def get_args():
    parser = argparse.ArgumentParser(description="labeler training script")
    parser.add_argument('--name', type=str)  # name of the experiment
    parser.add_argument('--project', type=str, help='Project name for wandb')

    parser.add_argument("--imagenet_root",
                        type=str,
                        default="<DATA_ROOT>/dataset/imagenet",
                        help="Path to the ImageNet dataset root directory"
                        )
    parser.add_argument('--class_file', type=str, default="LOC_synset_mapping.txt", help='class file')
    parser.add_argument("--num_workers",
                        type=int,
                        default=8,
                        help="Number of workers for parallel processing"
                        )
    parser.add_argument('--save_dir', type=str, default='./checkpoints/labelerIN1k',)
    # backbone
    parser.add_argument('--dino_arch', type=str, default='vitl16',
                        choices=['vits16', 'vits16plus', 'vitb16', 'vitl16', 'vith16plus', 'vit7b16'],
                        help='architecture of the DINO backbone; same as in dino repo')
    # head arch
    parser.add_argument('--arch', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='architecture of the head')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--norm', default=None, choices=['layernorm', 'batchnorm', None],
                        help='normalization type to use in the head')
    parser.add_argument('--apply_input_norm', action='store_true', help='l2 normalize input features')
    parser.add_argument('--apply_output_norm', action='store_true', help='l2 normalize output features')
    # MLP
    parser.add_argument('--hidden_dim', default=1024, type=int, help='dimension of hidden layers')
    parser.add_argument('--num_layers', default=2, type=int, help='number of hidden layers in MLP')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='optimizer to use for training')
    parser.add_argument('--lr', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    # scheduler
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'warmup_cosine'],
                        help='scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # training config
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size')
    parser.add_argument('--eval_interval', default=1, type=int, help='interval of epochs to evaluate the model')
    parser.add_argument('--used_feat', type=str, default='patch', choices=['patch'],
                        # choices=['patch', 'q', 'k', 'v'],
                        help='which feature to use for training, patch, q, k, v')

    # augmentation
    parser.add_argument('--size', type=int, default=512, help='input image size')
    parser.add_argument('--train_backbone', action='store_true', help='finetune entire model including backbone')
    parser.add_argument('--center_crop', action='store_true', help='use center crop for validation')

    # Resume training
    parser.add_argument('--resume_from', type=str, default=None, help='path to checkpoint to resume from')

    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    ## DDP args
    parser.add_argument("--patience", type=int, default=10000, help="Early stopping patience (epochs)")
    parser.add_argument('--real_label_path', type=str, default='<DATA_ROOT>/dataset/imagenet/real.json', help='path to real label file')

    #dinov3
    parser.add_argument('--dinov3_repo_dir', type=str, default='<DINOV3_REPO_PATH>', help='path to dinov3 repo')
    parser.add_argument('--patch_dropout', type=float, default=0.2, help='patch dropout rate')
    parser.add_argument('--pool_first', action='store_true', help="If set, pool features before classification head; else classify each patch then pool logits")
    parser.add_argument('--seg_annotation', type=str, default="<MASKCUT_DATASET_ROOT>/maskcut_imagenet_train_annotation.json", help='path to segmentation annotation file for maskcut dataset')
    parser.add_argument('--mask_label_dict', type=str, default="<MASKCUT_DATASET_ROOT>/maskcut_imagenet_mask_label_dict.pt", help='path to mask-label dict for maskcut dataset')
    parser.add_argument('--threshold', type=float, default=0.75, help='confidence threshold for ReLabel')
    parser.add_argument('--force_gt_label',  action='store_true', help='If set, always include gt label in training')

    args = parser.parse_args()
    return args


def evaluate_ddp(backbone, head, dataloader, device, args, return_predictions=False, local_rank=-1, world_size=4, forward_fn=get_masked_prediction, real_val=None):
    backbone.eval()
    head.eval()
    local_y_true = []
    local_y_pred = []
    local_index = []

    if local_rank == 0:
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    else:
        progress_bar = dataloader

    for batch_data in progress_bar:
        img = batch_data['image'].to(device)
        label = batch_data['label']
        index = batch_data['index']
        mask = None

        with torch.no_grad():
            pred = forward_fn(backbone, img, classification_head=head, mask=mask, pool_first=args.pool_first)

        local_y_true.extend(label.numpy())
        cur_y_pred = torch.argmax(pred, dim=1).cpu().numpy()
        local_y_pred.extend(cur_y_pred)
        local_index.extend(index.numpy())
    local_y_true = np.array(local_y_true)
    local_y_pred = np.array(local_y_pred)
    local_index = np.array(local_index)

    # gather everything on all ranks
    gathered_true = [None for _ in range(world_size)]
    gathered_pred = [None for _ in range(world_size)]
    gathered_index = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_true, local_y_true)
    dist.all_gather_object(gathered_pred, local_y_pred)
    dist.all_gather_object(gathered_index, local_index)

    y_true_all = np.concatenate(gathered_true, axis=0)
    y_pred_all = np.concatenate(gathered_pred, axis=0)
    index_all = np.concatenate(gathered_index, axis=0)


    if local_rank == 0:
        acc = accuracy_score(y_true_all, y_pred_all)

        y_pred_ordered_by_index = y_pred_all[np.argsort(index_all)]
        is_correct = [pred in real_val[i] for i, pred in enumerate(y_pred_ordered_by_index) if real_val[i]]
        real_acc = np.mean(is_correct)
    else:
        acc = None
        real_acc = None
    result = {
        'acc': acc,
        'real_acc': real_acc,
    }
    if return_predictions:
        return result, y_pred_all
    return result



def get_exp_name(args):
    # auto generate experiment name
    if args.arch == 'mlp':
        arch = f'mlp@{args.hidden_dim}x{args.num_layers}'
    else:
        arch = args.arch
    name = (f"DINOMaskCutRelabel@{args.dino_arch}_{arch}_dropout{args.dropout}_size{args.size}"
            f"_poolFirst{args.pool_first}_threshold{args.threshold}_forceGTlabel{args.force_gt_label}_op{args.optimizer}_lr{args.lr}ep{args.epochs}")
    return name

def train_step(
    backbone,
    head,
    batch_data,
    optimizer,
    device,
    criterion,
    experiment,
    train_backbone,          # kept for API compatibility; backbone should already be frozen
    scaler,
    forward_fn,              # get_masked_prediction wrapper
    patch_dropout,
    args
):
    img   = batch_data['image'].to(device, non_blocking=True)
    label = batch_data['label'].to(device, non_blocking=True)
    mask  = batch_data['mask'].to(device, non_blocking=True)

    # --- 0) Build "valid" sample mask: a sample is valid if its (pre-dropout) mask has any True ---
    # Works for (B, H, W) or (B, N). For int/float masks, >0 is treated as True.
    if mask.dim() == 3:
        valid_mask = (mask != 0).flatten(1).any(dim=1)   # (B,)
    elif mask.dim() == 2:
        valid_mask = (mask != 0).any(dim=1)              # (B,)
    else:
        raise ValueError(f"`mask` must be (B,H,W) or (B,N), got {tuple(mask.shape)}")

    num_valid = int(valid_mask.sum().item())

    # --- 1) Early exit if no valid samples (avoid NaNs/hangs in DDP) ---
    if num_valid == 0:
        if experiment is not None:
            experiment.log({
                'train/loss': 0.0,
                'train/valid_frac': 0.0,
                'train/filtered': img.shape[0],
                'train/lr': optimizer.param_groups[0]['lr'],
            })
        return {
        "loss": 0.0,
        "acc": 0.0,
        "hit_count": 0,
        "batch_size": 0,
    }

    # --- 2) Filter batch to valid samples only ---
    img_v   = img[valid_mask]
    label_v = label[valid_mask]
    mask_v  = mask[valid_mask]

    optimizer.zero_grad(set_to_none=True)

    # --- 3) Forward & loss (only valid samples participate in gradients) ---
    with torch.cuda.amp.autocast():
        pred_v = forward_fn(
            backbone,
            img_v,
            classification_head=head,
            mask=mask_v,
            pool_first=args.pool_first,
            random_dropout=patch_dropout
        )
        # criterion can be CE (labels: (B,)) or BCEWithLogits (labels: (B,K))
        total_loss = criterion(pred_v, label_v)

    # --- 4) Backward & step ---
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    batch_size = label.size(0)

    hit_count = (pred_v.argmax(dim=1) == label_v).sum().item()
    acc = hit_count / batch_size

    # --- 5) Logging ---
    if experiment is not None:
        B_total = img.shape[0]
        experiment.log({
            'train/loss': float(total_loss.item()),
            'train/lr': optimizer.param_groups[0]['lr'],
            'train/valid_frac': num_valid / float(B_total),
            'train/filtered': B_total - num_valid,
            'train/batch_acc': acc,
        })

    return {
        "loss": total_loss.item(),
        "acc": acc,
        "hit_count": hit_count,
        "batch_size": batch_size,
    }


def train_epoch(backbone, head, dataloader, optimizer, device, criterion, scheduler, experiment, train_backbone, scaler, local_rank, forward_fn, patch_dropout, args):
    fetch_data_time = 0.0
    train_time = 0.0


    batch_start_time = time.time()
    backbone.eval()
    head.train()

    total_loss = 0.0
    total_hit = 0
    total_samples = 0

    # Wrap dataloader with tqdm
    if local_rank == 0:
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
    else:
        progress_bar = dataloader

    experiment = None if local_rank != 0 else experiment  # Only log in the main process

    start_time = time.time()
    for step, batch_data in enumerate(progress_bar):
        fetch_data_time += time.time() - start_time
        start_time = time.time()
        batch_metrics = train_step(backbone, head, batch_data, optimizer, device, criterion, experiment, train_backbone, scaler, forward_fn, patch_dropout, args)
        if scheduler is not None:
            scheduler.step()
        total_loss += batch_metrics['loss']
        total_hit += batch_metrics['hit_count']
        total_samples += batch_metrics['batch_size']
        avg_loss = total_loss / (step + 1)
        if local_rank == 0:
            # Update tqdm description with current average loss
            progress_bar.set_description(f"Training - Loss: {avg_loss:.4f}")
        train_time += time.time() - start_time
        start_time = time.time()
    batch_end_time = time.time()
    if local_rank == 0:
        print(f"Epoch time: {batch_end_time - batch_start_time:.2f} seconds")
        print(f"Fetch data time: {fetch_data_time:.2f} seconds")
        print(f"Train time: {train_time:.2f} seconds")
        time_dict = {
            'train/fetch_data_time': fetch_data_time,
            'train/train_time': train_time,
        }
        experiment.log(time_dict)
    return total_loss / len(dataloader), total_hit / total_samples




def main():
    # Parse args
    args = get_args()
    assert args.train_backbone == False, "Currently only support training head"
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank

    seed_everything(args.seed)

    ## DDP init
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')


    # model
    backbone, feat_dim = get_dinov3_models(args.dino_arch, repo_dir=args.dinov3_repo_dir)
    forward_fn = get_masked_prediction
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False  # hard-freeze backbone

    classification_head = get_cls_head(
        arch=args.arch,
        input_dim=feat_dim,
        output_dim=1000,
        args=args
    )
    backbone.to(device)
    classification_head.to(device)
    
    # Load checkpoint if resuming
    start_epoch = 0
    checkpoint_loaded = False
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from}")
        
        if args.local_rank == 0:
            print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        
        # Load model states with DDP handling
        if args.train_backbone and checkpoint['backbone_state_dict'] is not None:
            load_state_dict_with_ddp_handling(backbone, checkpoint['backbone_state_dict'], verbose=(args.local_rank == 0))
            if args.local_rank == 0:
                print("Successfully loaded backbone state dict")
        load_state_dict_with_ddp_handling(classification_head, checkpoint['head_state_dict'], verbose=(args.local_rank == 0))
        
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        checkpoint_loaded = True
        
        if args.local_rank == 0:
            print(f"Successfully loaded classification head state dict")
            print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
    
    # DDP wrap the model
    if args.train_backbone:
        backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[args.local_rank], find_unused_parameters=True)
    classification_head = nn.parallel.DistributedDataParallel(classification_head, device_ids=[args.local_rank])

    # dataset
    with open(args.seg_annotation, 'r') as f:
        seg_annotation = json.load(f)
    mask_label_dict = torch.load(args.mask_label_dict, weights_only=False)
    helper = ImageNetHelper(root_dir=args.imagenet_root, class_file=args.class_file)
    train_transform = get_mask_augmentation(crop_size=args.size)
    val_transform = imagenet_transform_cpu(size=args.size, split='val', center_crop=args.center_crop)
    mask_size = args.size // 16
    train_set = RelabelMaskcutDataset(seg_annotation=seg_annotation, mask_label_dict=mask_label_dict, threshold=args.threshold, force_gt_label=args.force_gt_label, transform=train_transform, helper=helper, mask_size=mask_size)
    
    val_set = ReaLValSet(root=os.path.join(args.imagenet_root, 'val'), transform=val_transform)

    with open(args.real_label_path, 'r') as f:
        real_val = json.load(f)


    # DDP sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)


    # optimizer
    if args.train_backbone:
        models = [backbone, classification_head]
        optimizer = get_optimizer(models=models, args=args)
    else:
        optimizer = get_optimizer(models=classification_head, args=args)

    # scheduler
    args.iters_per_epoch = len(train_loader)
    scheduler = get_stepwise_scheduler(optimizer, args)
    
    # Load optimizer and scheduler states if resuming
    if checkpoint_loaded:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if args.local_rank == 0:
            print("Loaded optimizer and scheduler states from checkpoint")

    # mixed_precision training
    scaler = torch.cuda.amp.GradScaler()

    # loss function (cross entropy)
    criterion = nn.CrossEntropyLoss()

    # get experiment name
    if args.name is None:
        args.name = get_exp_name(args)

    # wandb config
    if args.local_rank == 0:
        resume_info = " (RESUMED)" if checkpoint_loaded else ""
        experiment = wandb.init(
            project=args.project,
            name=args.name,
            entity='<YOUR_WANDB_ENTITY>',
        )
        wandb.config.update(args)
        if not os.path.exists(os.path.join(args.save_dir, args.name)):
            os.makedirs(os.path.join(args.save_dir, args.name))
        es = EarlyStoppingDDP(patience=args.patience, mode='max')
        print(f"Starting experiment: {args.name}{resume_info}")
    else:
        experiment = None

    # tensor used to broadcast the stop-flag every epoch
    stop_tensor = torch.zeros(1, device=device)  # 0 = keep going

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.train_backbone:
            backbone.train()
        classification_head.train()
        train_loss, train_acc = train_epoch(backbone, classification_head, train_loader, optimizer, device, criterion, scheduler, experiment, train_backbone=args.train_backbone, scaler=scaler, local_rank=args.local_rank, forward_fn=forward_fn, patch_dropout=args.patch_dropout, args=args)
        if args.local_rank == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        y_pred = None

        if (epoch == 0) or ((epoch + 1) % args.eval_interval == 0) or (epoch == args.epochs - 1):
            # Evaluation
            eval_start_time = time.time()
            val_sampler.set_epoch(epoch)
            evaluation_metrics, y_pred = evaluate_ddp(backbone, classification_head, val_loader, device, return_predictions=True, local_rank=dist.get_rank(), world_size=dist.get_world_size(), forward_fn=forward_fn, real_val=real_val, args=args)
            # handle early stopping
            if args.local_rank == 0:
                eval_acc = evaluation_metrics['real_acc']
                es.step(eval_acc)
                if es.should_stop:
                    print(f"[rank-0] Early stop triggered at epoch {epoch}.")
                    stop_tensor.fill_(1)  # signal stop
            dist.broadcast(stop_tensor, src=0)

            # log evaluation metrics
            if args.local_rank == 0:
                eval_dict = {f'val/{key}': value for key, value in evaluation_metrics.items()}
                experiment.log(eval_dict)
                if (epoch == args.epochs - 1) or es.should_stop:
                    # log final performance
                    eval_dict = {f"final_val/{key}": value for key, value in evaluation_metrics.items()}
                    experiment.log(eval_dict)
                eval_end_time = time.time()
                print(f"Epoch [{epoch + 1}/{args.epochs}] Evaluation - Time: {eval_end_time - eval_start_time:.2f} seconds, Metrics: {evaluation_metrics}")

        # save checkpoint
        if args.local_rank == 0 and es.num_bad == 0:
            # Save for resume
            checkpoint = {
                'backbone_state_dict': backbone.state_dict() if args.train_backbone else None,
                'head_state_dict': classification_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'final_val_pred': y_pred,
                'args': args,
            }
            if args.scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, os.path.join(args.save_dir, args.name, 'checkpoint.pth'))
        dist.barrier()
        # Check if we should stop training
        if stop_tensor.item() == 1:
            break  # all ranks exit the epoch loop together

    # Finish wandb run
    if args.local_rank == 0:
        experiment.finish()

    # Cleanup DDP
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
