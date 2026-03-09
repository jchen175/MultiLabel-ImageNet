import pandas as pd
import os
import torch
import torch.distributed as dist
import wandb
import argparse

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from util import (update_pred_gt, load_state_dict_with_ddp_handling, Ralloss, AsymmetricLossOptimized,
                  smooth_targets_cap, str2bool, build_filename_to_label, get_label_mapping_from_json,
                  get_label_mapping_from_pt, MultiLabelImageFolder, ReaLValSet)

def get_args():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--arch", type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--pretrained', type=str, default='false', help='whether to use pretrained backbone; call str2bool as a workaround for sweep')
    parser.add_argument('--finetune', type=str,default=None, help='path to finetune checkpoint (optional)')
    # data paths
    parser.add_argument("--data_dir", type=str, default="<DATA_ROOT>/dataset/imagenet/")
    parser.add_argument("--relabel_csv", type=str, default="<DATA_ROOT>/dataset/imagenet/relabel_results/relabel_trainset.csv")
    parser.add_argument("--relabel_json", type=str, default=None, help="preprocessed relabel json file (optional)")
    parser.add_argument("--real_path", type=str, default="<DATA_ROOT>/dataset/imagenet/real.json")
    parser.add_argument("--save_dir", type=str, default="./ckpt/train_w_relabel")
    # relabeling args
    parser.add_argument("--scheme", type=str, choices=["one_hot", "relabel_prob", "relabel_thresh"], default="one_hot")
    parser.add_argument("--include_gt", type=str, default='false', help='whether to force-include GT label; call str2bool as a workaround for sweep')
    parser.add_argument('--include_pred_gt', type=str, default='false', help='whether to force-include predicted global label')
    parser.add_argument('--pred_gt_json', type=str, default=None,
                        help='path to json file containing predicted global labels; used only if --include_pred_gt is true')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--debug', action='store_true')

    # training hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "reduce_on_plateau", 'cosine'])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument('--patience', type=int, default=8, help='reduce_on_plateau patience')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR step size (epochs)')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR/reduce_op gamma')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='positive weight for bce logit loss')
    parser.add_argument('--label_smoothing_min', type=float, default=0.0, help='label smoothing for negative class')
    parser.add_argument('--label_smoothing_max', type=float, default=1.0, help='label smoothing for positive class')
    parser.add_argument('--init_bias', type=str, default='false', help='whether to initialize bias; call str2bool as a workaround for sweep')
    parser.add_argument('--nodecay', type=str, default='false',
                        help='whether to ignore weight decay on the classification head; call str2bool as a workaround for sweep')
    parser.add_argument('--name', type=str, default=None, help='name of experiment')

    # alternative loss
    parser.add_argument('--loss', type=str, choices=['asl', 'bce', 'ral'], default='bce')
    parser.add_argument('--gamma_neg', type=float, default=4, help='used only if --loss asl/ral')
    parser.add_argument('--gamma_pos', type=float, default=1, help='used only if --loss asl/ral')
    parser.add_argument('--clip', type=float, default=0.05, help='used only if --loss asl/ral')
    parser.add_argument('--resume_from', type=str, default=None, help='path to checkpoint to resume from')
    return parser.parse_args()

def get_exp_name(args):
    if args.loss == 'bce':
        loss_prefix = f"BCE-LSM@{args.label_smoothing_min}-{args.label_smoothing_max}-PW@{args.pos_weight}"
    elif args.loss == 'asl' or args.loss=='ral':
        loss_prefix = f"{args.loss.upper()}@gn{args.gamma_neg}-gp{args.gamma_pos}-c{args.clip}"
        if args.label_smoothing_min != 0 or args.label_smoothing_max != 1.0:
            print("Warning: label smoothing is not typically used with ASL loss.")
            print("Overwriting with 0 and 1.0")
            args.label_smoothing_min = 0.0
            args.label_smoothing_max = 1.0
    if args.scheme == 'one_hot':
        scheme = "OH"
    elif args.scheme == 'relabel_prob':
        scheme = "RProb"
    elif args.scheme == 'relabel_thresh':
        scheme = f"RThresh@{args.threshold}"
    if args.include_pred_gt and args.pred_gt_json is not None:
        if 'poolfirsttrue' in args.pred_gt_json.lower():
            pool_info = 'PoolFirstTrue'
        else:
            pool_info = 'PoolFirstFalse'
    else:
        pool_info = ''

    name = (f"{scheme}_forceGT{args.include_gt}_forcePredGT{args.include_pred_gt}_{pool_info}"
            f"{loss_prefix}_"
            f"lr@{args.lr}_Gbs@{args.global_batch_size}_op@{args.optimizer}WD@{args.weight_decay}NoDHead@{args.nodecay}_"
            f"_sche@{args.scheduler}_ep@{args.num_epochs}_{args.arch}-preT{args.pretrained}")
    return name

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
    args.init_bias = str2bool(args.init_bias)
    args.nodecay = str2bool(args.nodecay)
    args.include_gt = str2bool(args.include_gt)
    args.include_pred_gt = str2bool(args.include_pred_gt)
    args.pretrained = str2bool(args.pretrained)

    if not os.path.exists(args.data_dir):
        # override with env var if exists
        args.data_dir = os.environ['IMAGENET_DIR']
        args.real_path = os.environ['REAL_PATH']
        args.relabel_csv = os.environ['RELABEL_CSV']
        try:
            args.relabel_json = os.environ['RELABEL_JSON']
        except KeyError:
            args.relabel_json = None
        print(f"Using paths from env var...")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.name is None:
        args.name = get_exp_name(args)

    # Initialize wandb on rank 0 for experiment tracking
    if rank == 0:
        wandb.init(project="multilabel-classification", entity='<YOUR_WANDB_ENTITY>', name=args.name)
        wandb.config.update(args)

    if args.relabel_json is not None:
        collect_fn = get_label_mapping_from_pt if args.relabel_json.endswith('.pt') else get_label_mapping_from_json
        label_mapping = collect_fn(
            json_path=args.relabel_json,
            scheme=args.scheme,
            include_gt=args.include_gt,
            threshold=args.threshold,
            verbose=(rank == 0)
        )
    else:
        # Build label mapping from relabel CSV
        relabel_trainset = pd.read_csv(args.relabel_csv)
        label_mapping = build_filename_to_label(
            df=relabel_trainset,
            scheme=args.scheme,
            include_gt=args.include_gt,
            threshold=args.threshold,
            verbose=(rank == 0)
        )
    if args.include_pred_gt:
        if args.pred_gt_json is None:
            raise ValueError("--pred_gt_json must be provided if --include_pred_gt is true")
        label_mapping = update_pred_gt(label_mapping, args.pred_gt_json, verbose=(rank==0))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MultiLabelImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform,
                                         label_mapping=label_mapping)
    val_dataset = ReaLValSet(root=os.path.join(args.data_dir, "val"), transform=val_transform, real_path=args.real_path)
    num_classes = len(train_dataset.classes)  # number of classes inferred from ImageFolder

    # Distributed samplers for multi-GPU training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # DataLoaders (note: pin_memory=True for faster host-to-GPU transfers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    if args.arch == 'resnet50':
        model = models.resnet50(weights=None if args.pretrained==False else models.ResNet50_Weights.IMAGENET1K_V2)
    elif args.arch == 'resnet101':
        model = models.resnet101(weights=None if args.pretrained==False else models.ResNet101_Weights.IMAGENET1K_V2)
    # If our dataset has a different number of classes than ImageNet, replace the final layer
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    if args.init_bias:
        if args.label_smoothing_min > 0.0:
            prior = args.label_smoothing_min
        else:
            prior = 1/1000  # 1000 classes
        init_p = torch.tensor(prior)
        prior = torch.log(init_p/(1-init_p))
        with torch.no_grad():
            dtype_ = model.fc.bias.dtype
            size_ = model.fc.bias.size()
            prior_vector = torch.full(size_, prior, dtype=dtype_)
            model.fc.bias.copy_(prior_vector)
    model.to(device)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.finetune is not None:
        # load checkpoint for finetuning
        if not os.path.exists(args.finetune):
            raise FileNotFoundError(f"Finetune checkpoint file not found: {args.finetune}")

        if local_rank == 0:
            print(f"Loading finetune checkpoint from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location=device, weights_only=False)
        load_state_dict_with_ddp_handling(model, checkpoint['model_state_dict'],
                                          verbose=(local_rank == 0))
        if local_rank == 0:
            print(f"Successfully loaded classification head state dict from finetune checkpoint")

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
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if args.loss == 'bce':
        # Loss function for multi-label (Binary Cross-Entropy with logits)
        if args.pos_weight != 1.0:
            # pad to 1000-dim
            pos_weight = torch.ones(1000, device=device)
            pos_weight *= args.pos_weight
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == 'asl':
        criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip)
    elif args.loss == 'ral':
        criterion = Ralloss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.clip)

    else:
        raise ValueError(f"Unknown loss {args.loss}")

    # Optimizer (Adam used here; includes weight decay for regularization)
    if args.nodecay:
        if args.optimizer.lower() == 'sgd':
            for name, param in model.named_parameters():
                params = []
                if 'fc' in name:
                    params.append({'params': param, 'weight_decay': 0.0})
                else:
                    params.append({'params': param, 'weight_decay': args.weight_decay})
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer.lower() == 'adamw':
            params = []
            for name, param in model.named_parameters():
                if 'fc' in name:
                    params.append({'params': param, 'weight_decay': 0.0})
                else:
                    params.append({'params': param, 'weight_decay': args.weight_decay})
            optimizer = torch.optim.AdamW(params, lr=args.lr)
    else:
        if args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler: Reduce LR on Plateau (will be combined with manual warmup)
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

    for epoch in range(start_epoch, args.num_epochs):
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
            # After warmup, use the ReduceLROnPlateau scheduler (monitoring validation loss)
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

        # ---- Training Phase ----
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss_sum = 0.0
        train_samples_count = 0

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} - Training...")
            iterator = tqdm(train_loader)
        else:
            iterator = train_loader

        for batch_step, (images, main_labels, targets) in enumerate(iterator):
            images = images.to(device)
            targets = targets.to(device).float()

            # Forward pass with autocast for mixed precision
            with autocast():
                logits = model(images)  # [B, 1000] logits

                # ---- Filter out empty (all-zero) targets ----
                valid_mask = (targets.sum(dim=1) > 0)  # [B]
                n_valid = int(valid_mask.sum().item())

                if n_valid > 0:
                    images_v = images[valid_mask]
                    targets_v = targets[valid_mask]

                    logits_v = logits[valid_mask]
                    if args.label_smoothing_min > 0.0 or args.label_smoothing_max < 1.0:
                        targets_v = smooth_targets_cap(targets_v, pos_val=args.label_smoothing_max, neg_val=args.label_smoothing_min)
                    loss = criterion(logits_v, targets_v)  # BCEWithLogitsLoss, averaged over valid samples

                    # Stats use only valid samples
                    train_loss_sum += loss.item() * n_valid
                    train_samples_count += n_valid
                else:
                    # No valid samples on this rank: create a zero loss that keeps the graph consistent for DDP
                    loss = logits.sum() * 0.0
                    # No contribution to running loss/sample counters when nothing valid here

            # Backprop + step with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                # Show the loss for *this rank*; if no valid samples here, it's 0.0
                iterator.set_description(f"Train Loss: {loss.item():.4f}")
                wandb.log({
                    "Train/Train Loss": float(loss.item()),
                    "Train/Learning Rate": optimizer.param_groups[0]['lr'],
                    "Train/Valid Samples This Batch": n_valid,
                    "Train/GradScaler Scale": scaler.get_scale(),
                })

            if args.debug and batch_step >= 10:
                break

        dist.barrier()  # ensure all processes have finished training phase

        # Compute average training loss for this epoch on **this process**
        avg_train_loss = train_loss_sum / train_samples_count if train_samples_count > 0 else 0.0

        # ---- Validation Phase ----
        model.eval()
        val_loss_sum = 0.0
        val_samples_count = 0
        correct_single = 0  # single-label correct predictions count
        recall_single_hits = 0

        total_tp = total_fp = total_fn = 0  # counters for multi-label metrics
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} - Validating...")
            iterator = tqdm(val_loader)
        else:
            iterator = val_loader

        with torch.no_grad():
            for batch in iterator:
                images = batch['image'].to(device)
                main_labels = batch['label'].to(device).long()
                is_null = batch['is_null']
                batch_n = images.size(0)
                targets = batch['multi_label'].to(device)

                # Forward pass with autocast for mixed precision
                with autocast():
                    logits = model(images)
                    val_loss = criterion(logits, targets)

                # Accumulate loss & sample count (for averaging later)
                val_loss_sum += val_loss.item() * batch_n
                val_samples_count += batch_n

                # ---- Single-label metrics ----
                # Accuracy: argmax over logits vs GT
                preds_single = torch.argmax(logits, dim=1)
                correct_single += (preds_single == main_labels).sum().item()

                # Recall (single-label): is GT included in the predicted multi-label set?
                probs = torch.sigmoid(logits)
                preds_multi = (probs >= 0.5)  # bool tensor [B, C]
                # gather returns [B,1]; squeeze to [B]; sum counts how many GTs were included
                recall_single_hits += preds_multi.gather(1, main_labels.view(-1, 1)).squeeze(1).sum().item()

                # ---- Multi-label metrics (exclude ReaL nulls) ----
                # Convert is_null -> bool tensor on device; valid rows are ~is_null
                if isinstance(is_null, torch.Tensor):
                    null_mask = is_null.to(device).bool()
                else:
                    null_mask = torch.as_tensor(is_null, device=device, dtype=torch.bool)
                valid_mask = ~null_mask

                if valid_mask.any():
                    # Filter to non-null rows only
                    pm = preds_multi[valid_mask]  # bool [Bv, C]
                    tm = (targets[valid_mask] > 0.5)  # bool [Bv, C]

                    # Micro TP/FP/FN
                    total_tp += int((pm & tm).sum().item())
                    total_fp += int((pm & ~tm).sum().item())
                    total_fn += int((~pm & tm).sum().item())

        dist.barrier()  # ensure all processes have finished validation

        # ---- Aggregation of Metrics (across GPUs) ----
        metrics_tensor = torch.tensor([
            correct_single, val_samples_count, total_tp, total_fp, total_fn,
            train_loss_sum, train_samples_count, val_loss_sum, val_samples_count,
            recall_single_hits
        ], device=device, dtype=torch.float64)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        # Extract aggregated values
        total_correct_single = int(metrics_tensor[0].item())
        total_val_samples = int(metrics_tensor[1].item())
        total_tp_all = int(metrics_tensor[2].item())
        total_fp_all = int(metrics_tensor[3].item())
        total_fn_all = int(metrics_tensor[4].item())
        total_train_loss_sum = metrics_tensor[5].item()
        total_train_count = int(metrics_tensor[6].item())
        total_val_loss_sum = metrics_tensor[7].item()
        total_val_count = int(metrics_tensor[8].item())
        total_recall_single_hits = int(metrics_tensor[9].item())  # <--- NEW
        # Compute global average losses
        global_train_loss = total_train_loss_sum / total_train_count if total_train_count > 0 else 0.0
        global_val_loss = total_val_loss_sum / total_val_count if total_val_count > 0 else 0.0
        # Compute accuracy and multi-label metrics
        val_accuracy = total_correct_single / max(total_val_samples, 1)
        val_prec_micro = total_tp_all / max(total_tp_all + total_fp_all, 1)
        val_recall_micro = total_tp_all / max(total_tp_all + total_fn_all, 1)
        val_f1_micro = (2 * val_prec_micro * val_recall_micro /
                        max(val_prec_micro + val_recall_micro, 1e-12))
        val_recall_single = total_recall_single_hits / max(total_val_samples, 1)

        # ---- Logging (only on rank 0) ----
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} - "
                  f"Train Loss: {global_train_loss:.4f}, Val Loss: {global_val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}, Precision: {val_prec_micro:.4f}, Recall: {val_recall_micro:.4f}, F1: {val_f1_micro:.4f}")
            results = {
                "epoch": epoch + 1,
                "train_loss": global_train_loss,
                "val_loss": global_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision_micro": val_prec_micro,
                "val_recall_micro": val_recall_micro,
                "val_f1_micro": val_f1_micro,
                "val_recall_single": val_recall_single,  # <--- NEW
            }
            wandb.log(results)
            # save checkpoint
            checkpoint_path = f"{args.name}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, os.path.join(args.save_dir, checkpoint_path))
        dist.barrier()
        # ... (scheduler step handled below) ...
        if args.debug and epoch >= 2:
            break
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()