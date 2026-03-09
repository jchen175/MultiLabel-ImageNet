from tqdm import tqdm
import os
import argparse
import torch
import json
from collections import defaultdict
from pycocotools.mask import decode as coco_decode
from torch.nn import functional as F
import numpy as np
import cv2

def reconstruct_and_interpolate_label_map(m, size=(15, 15), mode='bilinear', resize=True):
    """
    Reconstructs a [1000, H, W] label map from top-5 logits and indices,
    and interpolates to a new size (m, n).

    Args:
        m (torch.Tensor): Tensor of shape [2, 5, H, W].
                          m[0] are logits, m[1] are class indices.
        size (tuple): Desired output size as (m, n).
        mode (str): Interpolation mode: 'bilinear' or 'nearest'.

    Returns:
        torch.Tensor: Interpolated label map of shape [1000, m, n].
    """
    logits = m[0]  # shape [5, H, W]
    indices = m[1]  # shape [5, H, W]

    C, H, W = 1000, logits.shape[1], logits.shape[2]

    # Initialize zero tensor for full logits
    full_logits = torch.zeros((C, H, W), dtype=logits.dtype, device=logits.device)

    # Scatter top-5 logits to full 1000-class map
    for i in range(5):
        index = indices[i]  # shape [H, W]
        logit = logits[i]  # shape [H, W]
        full_logits.scatter_(0, index.long().unsqueeze(0), logit.unsqueeze(0))

    if resize:
        # Interpolate to (m, n)
        full_logits = full_logits.unsqueeze(0)  # add batch dim: [1, 1000, H, W]
        interpolated = F.interpolate(full_logits, size=size, mode=mode, align_corners=(mode == 'bilinear'))
        return interpolated.squeeze(0)  # remove batch dim: [1000, m, n]
    else:
        return full_logits

def get_top5_prob_class(image_index, mcut_annotation, relabel_path, image_id2mask_id):
    cur_result = {}
    image_id = mcut_annotation['images'][image_index]['id']
    image_filename = mcut_annotation['images'][image_index]['file_name']

    masks = []
    mask_ids = image_id2mask_id[image_id]
    for i in mask_ids:
        cur_annotation = mcut_annotation['annotations'][i-1]
        assert cur_annotation['id'] == i
        masks.append(coco_decode(cur_annotation['segmentation']))
    if len(masks) == 0:
        return cur_result

    m = torch.load(os.path.join(relabel_path, image_filename.replace('.JPEG', '.pt')))
    m_reshaped = reconstruct_and_interpolate_label_map(m, mode='bilinear', resize=False)

    for mask_id, mask in zip(mask_ids, masks):
        mask = cv2.resize(mask.astype(np.float32), (15, 15), interpolation=cv2.INTER_LINEAR)
        logits = torch.sum(m_reshaped * mask, axis=(1,2)) / np.sum(mask)
        probs = F.softmax(logits, dim=0)
        prob, index = torch.topk(probs, 5)
        cur_result[mask_id] = {
            'prob': prob.cpu().numpy().tolist(),
            'label': index.cpu().numpy().tolist()
        }
    return cur_result

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--segmentation_file', type=str, default='<DATA_ROOT>/dataset/imagenet/maskcut_train_dinov2_vitg_s672_tau012.json', help="File containing maskcut segmentation information")
    args.add_argument("--imagenet_root", type=str, default="<DATA_ROOT>/dataset/imagenet/", help="Root directory of ImageNet dataset")
    args.add_argument("--class_file", type=str, default="LOC_synset_mapping.txt", help="File containing class mappings for ImageNet")
    args.add_argument("--relabel_root", type=str, default='<DATA_ROOT>/dataset/imagenet/ReLabel', help="Root directory for ReLabel's label maps")
    args.add_argument("--output_dir", type=str, default="<DATA_ROOT>/dataset/imagenet/relabel_result/mask_level_relabel", help="Directory to save the output predictions")

    # Parallel processing parameters
    args.add_argument('--total_jobs', type=int, default=8,
                        help='Total number of parallel jobs')
    args.add_argument('--job_index', type=int, default=0,
                        help='Current job index (0-based)')
    args.add_argument('--save_interval', type=int, default=2000,
                        help='Interval for saving intermediate results')

    args.add_argument('--merge_only', action='store_true',
                        help='If set, only merge existing results without processing new data')
    args = args.parse_args()

    output_dir = os.path.join(args.output_dir, '.'.join(os.path.basename(args.segmentation_file).split('.')[:-1]))

    if args.merge_only:
        merged_result = {}
        all_files = os.listdir(output_dir)
        for file in tqdm(all_files):
            if file.endswith('.pt'):
                file_path = os.path.join(output_dir, file)
                part_result = torch.load(file_path)
                merged_result.update(part_result)
        save_file = os.path.join(output_dir, f"mask_level_relabel_merged.pt")
        torch.save(merged_result, save_file)
        print(f"Merged results saved to {save_file}")
        print(f"Total masks processed: {len(merged_result)}")
        exit(0)

    os.makedirs(output_dir, exist_ok=True)

    with open(args.segmentation_file, 'r') as f:
        mcut_annotation = json.load(f)

    mask_id2image_id = {}
    for ann in tqdm(mcut_annotation['annotations']):
        mask_id2image_id[ann['id']] = ann['image_id']
    image_id2mask_id = defaultdict(list)
    for k, v in tqdm(mask_id2image_id.items()):
        image_id2mask_id[int(v)].append(k)

    all_masks = len(mcut_annotation['images'])
    per_job_load = int(np.ceil(all_masks / args.total_jobs))
    start_index = args.job_index * per_job_load
    end_index = min((args.job_index + 1) * per_job_load, len(mcut_annotation['images']))
    print(f"Processing job {args.job_index+1}/{args.total_jobs}, images {start_index} to {end_index} out of {len(mcut_annotation['images'])}")
    result = {}
    cur_start_index = start_index
    for index in tqdm(range(start_index, end_index), desc="Processing images"):
        cur_result = get_top5_prob_class(index, mcut_annotation, args.relabel_root, image_id2mask_id)
        result.update(cur_result)
        if len(result) >= args.save_interval:
            print(f"Saving intermediate results at image index {index}")
            save_file = os.path.join(output_dir, f"mask_level_relabel_{cur_start_index}_{index}.pt")
            torch.save(result, save_file)
            result = {}
            cur_start_index = index + 1

    save_file = os.path.join(output_dir, f"mask_level_relabel_{cur_start_index}_{end_index-1}.pt")
    torch.save(result, save_file)


