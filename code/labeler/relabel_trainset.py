import os
import argparse
from datasets.base_datasets import ImageNetHelper, MCutDataset, SubsetDataset, ImageFolderWithFilenames, split_dataset_indices
from models.pretrained_encoder import get_dinov2_models, get_dinov3_models
import torch
from torchvision import transforms
from tqdm import tqdm
from util import load_cls_head, get_masked_prediction, get_pooled_patch_features
"""
Relabel ImageNet training set using labeler head with maskcut masks.
Each output file (per job) is a dict: {
    'id': same as the mask
    'prob': top 5 probabilities
    'label': predicted label index for top 5
}
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Relabel training set with parallel processing')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', type=str, default='<DATA_ROOT>/<CKPT_PATH>/checkpoint.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--dino_version', type=str, default='dinov3',)
    parser.add_argument('--model_type', type=str, default='vitl16',
                        help='Model type for DINOv2/3 (e.g., vitl, vitb, vits)')
    
    # Data parameters
    parser.add_argument('--annotation_file', type=str, default='<DATA_ROOT>/dataset/imagenet/maskcut_train_dinov2_vitg_s672_tau012.json',
                        help='Path to the maskcut annotation file')
    parser.add_argument('--output_dir', type=str, default='<DATA_ROOT>/dataset/imagenet/relabel_results/maskcut_train_dinov2_vitg_s672_tau012',
                        help='Output directory for relabel results')
    parser.add_argument('--imagenet_root', type=str, default="<DATA_ROOT>/dataset/imagenet/")
    parser.add_argument('--repo_dir', type=str, default='<DATA_ROOT>/projects/semantic_ssl/dinov3',
                        help='Path to DINOv3 repo; used for loading models')
    parser.add_argument('--input_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loader workers')
    
    # Parallel processing parameters
    parser.add_argument('--total_jobs', type=int, default=8,
                        help='Total number of parallel jobs')
    parser.add_argument('--job_index', type=int, default=0,
                        help='Current job index (0-based)')
    
    # Processing parameters
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to save')
    parser.add_argument('--random_dropout', type=float, default=0.0,
                        help='Fraction of masked-in patches to remove')
    parser.add_argument('--global_pred', action='store_true', help='Global prediction without mask')
    parser.add_argument('--split', type=str, default='train', help='ImageNet split to use (train/val)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    # parse pool_first from checkpoint name
    if 'poolfirsttrue' in args.checkpoint_path.lower():
        args.pool_first = True
    elif 'poolfirstfalse' in args.checkpoint_path.lower():
        args.pool_first = False
    else:
        print("Warning: pool_first not specified in checkpoint path, using default True")
        args.pool_first = True  # default
    
    print(f"Starting job {args.job_index + 1}/{args.total_jobs}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Annotation file: {args.annotation_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Input size: {args.input_size}")
    
    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint_path, weights_only=False)
    head = load_cls_head(ckpt)
    if args.dino_version == 'dinov2':
        backbone = get_dinov2_models(args.model_type)[0]
    elif args.dino_version == 'dinov3':
        backbone = get_dinov3_models(args.model_type, repo_dir=args.repo_dir)[0]
    else:
        raise ValueError("dino_version must be 'dinov2' or 'dinov3'")
    del ckpt
    
    backbone.eval()
    head.eval()
    backbone.cuda()
    head.cuda()
    
    # Setup data preprocessing
    helper = ImageNetHelper(root_dir=args.imagenet_root)
    opener = transforms.Compose([
        transforms.Resize(size=(args.input_size, args.input_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    patch_size = 14 if args.dino_version == 'dinov2' else 16
    if not args.global_pred:
        mask_size = args.input_size // patch_size


        # Load full dataset
        print("Loading dataset...")
        full_dataset = MCutDataset(
            args.annotation_file,
            transform=opener,
            imagenet_helper=helper,
            mask_size=mask_size
        )

        # Split dataset for this job
        total_length = len(full_dataset)
        start_idx, end_idx = split_dataset_indices(total_length, args.total_jobs, args.job_index)

        print(f"Processing samples {start_idx} to {end_idx-1} (total: {end_idx - start_idx}) out of {total_length}")

        # Create subset dataset for this job
        subset_dataset = SubsetDataset(full_dataset, start_idx, end_idx)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Setup dataloader
        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers = True,
            prefetch_factor = 2
        )

        # Process data
        results = {}
        print("Processing batches...")
        for batch in tqdm(dataloader, desc=f"Job {args.job_index + 1}/{args.total_jobs}"):
            with torch.no_grad():
                if args.dino_version == 'dinov2':
                    pooled_feature = get_pooled_patch_features(
                        model=backbone,
                        x=batch['image'].cuda(),
                        patch_dim=1,  # index of the patch dimension in model output
                        mask=batch['mask'].cuda(),
                        random_dropout=args.random_dropout,  # fraction of masked-in patches to remove
                    )
                    logits = head(pooled_feature)
                else:
                    logits = get_masked_prediction(
                            model=backbone,
                        x=batch['image'].cuda(),
                            classification_head=head,
                            mask=batch['mask'].cuda(),
                            random_dropout = 0.0,  # fraction of *masked-in* patches to drop
                            pool_first = args.pool_first,  # True: pool features -> head; False: head -> pool logits
                    )
                probs = torch.softmax(logits, dim=-1)
                top_k_prob, top_k_label = torch.topk(probs, k=args.top_k, dim=-1)
                top_k_prob = top_k_prob.cpu().numpy()
                top_k_label = top_k_label.cpu().numpy()
                mask_ids = batch['mask_id'].numpy()

                for i in range(len(batch['mask_id'])):
                    results[mask_ids[i].item()] = {
                        'prob': top_k_prob[i].tolist(),
                        'label': top_k_label[i].tolist()
                    }

        # Save results with job index in filename
        output_filename = f"relabel_results_job_{args.job_index:03d}.pt"
        output_path = os.path.join(args.output_dir, output_filename)
        torch.save(results, output_path)
    else:
        # Load full dataset
        print("Loading dataset...")
        full_dataset = ImageFolderWithFilenames(root=os.path.join(args.imagenet_root, args.split), transform=opener)
        # Split dataset for this job
        total_length = len(full_dataset)
        start_idx, end_idx = split_dataset_indices(total_length, args.total_jobs, args.job_index)

        print(f"Processing samples {start_idx} to {end_idx - 1} (total: {end_idx - start_idx}) out of {total_length}")

        # Create subset dataset for this job
        subset_dataset = SubsetDataset(full_dataset, start_idx, end_idx)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Setup dataloader
        dataloader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=True,
            prefetch_factor=2
        )

        # Process data
        results = {}
        print("Processing batches...")
        for batch_image, batch_target, batch_filename in tqdm(dataloader, desc=f"Job {args.job_index + 1}/{args.total_jobs}"):
            with torch.no_grad():
                if args.dino_version == 'dinov2':
                    pooled_feature = get_pooled_patch_features(
                        model=backbone,
                        x=batch_image.cuda(),
                        patch_dim=1,  # index of the patch dimension in model output
                        mask=None,
                        random_dropout=args.random_dropout,  # fraction of masked-in patches to remove
                    )
                    logits = head(pooled_feature)
                else:
                    logits = get_masked_prediction(
                        model=backbone,
                        x=batch_image.cuda(),
                        classification_head=head,
                        mask=None,
                        random_dropout=0.0,  # fraction of *masked-in* patches to drop
                        pool_first=args.pool_first,  # True: pool features -> head; False: head -> pool logits
                    )
                probs = torch.softmax(logits, dim=-1)
                top_k_prob, top_k_label = torch.topk(probs, k=args.top_k, dim=-1)
                top_k_prob = top_k_prob.cpu().numpy()
                top_k_label = top_k_label.cpu().numpy()

                for i in range(len(batch_filename)):
                    results[batch_filename[i]] = {
                        'prob': top_k_prob[i].tolist(),
                        'label': top_k_label[i].tolist()
                    }

        # Save results with job index in filename
        output_filename = f"relabel_results_job_{args.job_index:03d}.pt"
        output_path = os.path.join(args.output_dir, output_filename)
        torch.save(results, output_path)
    
    print(f"Job {args.job_index + 1}/{args.total_jobs} completed!")
    print(f"Processed {len(results)} samples")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()






