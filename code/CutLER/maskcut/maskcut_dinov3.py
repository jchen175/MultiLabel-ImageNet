#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# git clone https://github.com/facebookresearch/CutLER.git
import os
import sys
sys.path.append('../')
import argparse
import numpy as np
from tqdm import tqdm
import re
import datetime
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh
import json
import dino
import wandb
from einops import rearrange

# modfied by Xudong Wang based on third_party/TokenCut
sys.path.append('../')
sys.path.append('../third_party')
from TokenCut.unsupervised_saliency_detection import utils, metric
from TokenCut.unsupervised_saliency_detection.object_discovery import detect_box
# bilateral_solver codes are modfied based on https://github.com/poolio/bilateral_solver/blob/master/notebooks/bilateral_solver.ipynb
# from TokenCut.unsupervised_saliency_detection.bilateral_solver import BilateralSolver, BilateralGrid
# crf codes are are modfied based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py
from crf import densecrf

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])


def get_dinov2_models(key):
    """
    Load DINOv2 models with registers.
    Returns:
        dict: A dictionary containing DINOv2 models.
    """
    assert key in ['vits', 'vitb', 'vitl', 'vitg']

    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{key}14_reg')
    size = model.forward_features(torch.randn(1, 3, 224, 224))['x_norm_patchtokens'].size()[-1]
    return model, size


def get_dinov3_models(key, repo_dir='<dinov3_repo_path>'):
    dinov3_loc = {
        'vits16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vits16plus': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vitb16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vitl16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vith16plus': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'vit7b16': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_tiny': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_small': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_base': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
        'convnext_large': '<apply_via_https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>',
    }
    model = torch.hub.load(repo_dir, f'dinov3_{key}', source='local', weights=dinov3_loc[key])
    size = model.forward_features(torch.randn(1, 3, 224, 224))['x_norm_patchtokens'].size()[-1]
    return model, size


class ViTLastLayerExtractor(torch.nn.Module):
    """
    Wraps a DINO-v2 ViT and exposes last-layer q / k / v or the
    normal patch tokens.  Output shape is always (B, T-5, embed_dim).
    """
    def __init__(self, base_vit, default="k", feat_last=True, require_grad=False):
        super().__init__()
        assert default in {"q", "k", "v", "patch"}
        self.base = base_vit
        self.default = default
        self.feat_last = feat_last
        self.require_grad = require_grad

        # ----- register one forward hook on *the very last* attention -----
        if getattr(self.base, "chunked_blocks", False):   # g-14 uses chunked_blocks
            self.last_block = self.base.blocks[-1][-1]
        else:
            self.last_block = self.base.blocks[-1]

        self._container = {}   # will hold q/k/v for the current forward pass

        def _hook(attn_mod, inputs, output):
            """
            inputs[0] : x  -> (B, T, embed_dim)
            Store q, k, v as (B, H, T, d) in self._container
            """
            x = inputs[0]
            B, T, C = x.shape
            qkv = attn_mod.qkv(x).reshape(
                B, T, 3, attn_mod.num_heads, C // attn_mod.num_heads
            ).permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
            self._container["q"], self._container["k"], self._container["v"] = qkv

        # keep the handle so the hook lives as long as the wrapper
        self._hook_handle = self.last_block.attn.register_forward_hook(_hook)

    # ---------------------------------------------------------------------

    # @torch.no_grad()          # remove if you need gradients

    def _forward(self, images, return_type=None):

        """
        return_type ∈ {"q", "k", "v", "patch"}; falls back to self.default.
        Output: (B, T-5, embed_dim)  where embed_dim = heads * head_dim
        """
        rtype = return_type or self.default
        assert rtype in {"q", "k", "v", "patch"}

        # run one pass – forward_features is enough, no classification head
        feats = self.base.forward_features(images)        # dict

        if rtype == "patch":
            x = feats["x_norm_patchtokens"]
            if not self.feat_last:
                x = rearrange(x, "b t d -> b d t")
            return x

        # otherwise take what the hook cached
        tensor = self._container[rtype]                   # (B, H, T, d)
        tensor = tensor[:, :, 5:, :]                      # remove first 5 tokens
        if self.feat_last:
            tensor = rearrange(tensor, "b h t d -> b (h d) t")
        else:
            tensor = rearrange(tensor, "b h t d -> b t (h d)")
        return tensor

    def forward(self, images, return_type=None):
        if not self.require_grad:
            with torch.no_grad():
                return self._forward(images, return_type)
        else:
            return self._forward(images, return_type)



def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix 
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def maskcut_forward(feats, dims, scales, init_image_size, tau, N, device):
    """
    Implementation of MaskCut.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      N: number of pseudo-masks per image.
    """
    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.from_numpy(np.zeros(dims))
            painting = painting.to(device)
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        pseudo_mask = pseudo_mask.to(device)
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                pseudo_mask = pseudo_mask.to(device)
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        bipartition_masked = bipartition.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        eigvec = eigvec.to(device)
        eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return seed, bipartitions, eigvecs


def maskcut(img_path, backbone, patch_size, tau, N, fixed_size, device) :
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    tensor = tensor.to(device)
    feat = backbone(tensor)[0]

    _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N, device=device)

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs, I_new


def maskcut_w_feature(feature_map, tau, N, fixed_size, device) :
    bipartitions, eigvecs = [], []
    feat_h = feat_w = int(feature_map.size()[1]**0.5)
    patch_size = fixed_size // feat_h
    w = h = fixed_size

    _, bipartition, eigvec = maskcut_forward(feature_map, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N, device=device)

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs



def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info


def get_dinov3(key, feat, size, repo_dir='<dinov3_repo_path>'):
    base_model, _ = get_dinov3_models(key, repo_dir)
    model = ViTLastLayerExtractor(base_model, default=feat)
    size = int(size)
    return model, 'dinov3_' + key + '_' + feat + f'_{size}', size



# necessay info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2023,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MaskCut script')
    # default arguments
    parser.add_argument('--out_dir', type=str, default="<DATA_ROOT>/dataset/imagenet/maskcut_train_temp_dinov2_vitg_k_896_tau075_nocrf/", help='output directory')
    parser.add_argument('--dino_version', type=str, default='dinov2', choices=['dino', 'dinov2', 'dinov3'], help='which architecture')
    parser.add_argument('--vit_arch', type=str, default='vitg', choices=['base', 'small', 'vits', 'vitb', 'vitl', 'vitg', 'vits16', 'vits16plus', 'vitb16', 'vitl16', 'vith16plus', 'vit7b16', 'vits14'], help='which architecture')
    parser.add_argument('--vit_feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv', 'patch'], help='which features')
    parser.add_argument('--patch_size', type=int, default=14, choices=[16, 8, 14], help='patch size')
    parser.add_argument('--nb_vis', type=int, default=20, choices=[1, 200], help='nb of visualization')
    parser.add_argument('--img_path', type=str, default=None, help='single image visualization')
    parser.add_argument('--dinov3_repo_dir', type=str, default='<dinov3_repo_path>', help='path to the dinov3 repo')

    # additional arguments
    parser.add_argument('--dataset_path', type=str, default="<DATA_ROOT>/dataset/imagenet/train/", help='path to the dataset')
    parser.add_argument('--tau', type=float, default=0.75, help='threshold used for producing binary graph')
    parser.add_argument('--num_folder_per_job', type=int, default=1, help='the number of folders each job processes')
    parser.add_argument('--job_index', type=int, default=0, help='the index of the job (for imagenet: in the range of 0 to 1000/args.num_folder_per_job-1)')
    parser.add_argument('--fixed_size', type=int, default=896, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=4, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--crf', type=str, default='false', help='whether to use crf post-processing')
    parser.add_argument('--split', type=str, default='train')

    args = parser.parse_args()
    # wandb config
    experiment = wandb.init(
    )
    config = wandb.config
    args.job_index = config.get("job_index", args.job_index)

    args.split = config.get("split", args.split)
    args.fixed_size = config.get("fixed_size", args.fixed_size)
    args.dino_version = config.get("dino_version", args.dino_version)
    args.vit_arch = config.get("vit_arch", args.vit_arch)
    args.vit_feat = config.get("vit_feat", args.vit_feat)
    args.patch_size = config.get("patch_size", args.patch_size)
    args.tau = config.get("tau", args.tau)
    args.N = config.get("N", args.N)
    args.dataset_path = config.get("dataset_path", args.dataset_path)
    args.crf = config.get("crf", args.crf)
    if args.crf.lower() in ['true', '1', 'yes']:
        args.crf = True
    else:
        args.crf = False

    tail = "" if 'v2' not in args.dataset_path else '_v2'
    args.out_dir = os.path.join(args.dataset_path, f"maskcut_{args.split}_temp_{args.dino_version}_{args.vit_arch}_{args.vit_feat}_{args.fixed_size}_tau{args.tau}_crf{args.crf}"+tail)
    args.dataset_path = os.path.join(args.dataset_path, args.split)
    wandb.config.update(vars(args))



    if args.dino_version == 'dino':
        if args.pretrain_path is not None:
            url = args.pretrain_path
        if args.vit_arch == 'base' and args.patch_size == 8:
            if args.pretrain_path is None:
                url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            feat_dim = 768
        elif args.vit_arch == 'small' and args.patch_size == 8:
            if args.pretrain_path is None:
                url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            feat_dim = 384

        backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    elif args.dino_version == 'dinov2':
        dino_model, feat_dim = get_dinov2_models(key = args.vit_arch)
        backbone = ViTLastLayerExtractor(dino_model, default=args.vit_feat,feat_last=True)
    elif args.dino_version == 'dinov3':
        backbone, key__, fixed_size__ = get_dinov3(args.vit_arch, args.vit_feat, args.fixed_size, args.dinov3_repo_dir)
    else:
        raise NotImplementedError('Only dino, dinov2 and dinov3 are supported!')





    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print (msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    img_folders = os.listdir(args.dataset_path)
    # remove non-folder files
    img_folders = [f for f in img_folders if os.path.isdir(os.path.join(args.dataset_path, f))]

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)

    start_idx = max(args.job_index*args.num_folder_per_job, 0)
    end_idx = min((args.job_index+1)*args.num_folder_per_job, len(img_folders))

    image_id, segmentation_id = 1, 1
    image_names = []
    for img_folder in img_folders[start_idx:end_idx]:
        args.img_dir = os.path.join(args.dataset_path, img_folder)
        img_list = sorted(os.listdir(args.img_dir))

        for img_name in tqdm(img_list) :
            # get image path
            img_path = os.path.join(args.img_dir, img_name)
            # get pseudo-masks for each image using MaskCut
            try:
                bipartitions, _, I_new = maskcut(img_path, backbone, args.patch_size, \
                    args.tau, N=args.N, fixed_size=args.fixed_size, device='cpu' if args.cpu else 'cuda')
            except:
                print(f'Skipping {img_name}')
                continue

            I = Image.open(img_path).convert('RGB')
            width, height = I.size
            for idx, bipartition in enumerate(bipartitions):
                if args.crf:
                    # post-process pesudo-masks with CRF
                    pseudo_mask = densecrf(np.array(I_new), bipartition)
                else:
                    pseudo_mask = bipartition.astype(np.float32)
                # post-process pesudo-masks with CRF

                pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

                # filter out the mask that have a very different pseudo-mask after the CRF
                mask1 = torch.from_numpy(bipartition)
                mask2 = torch.from_numpy(pseudo_mask)
                if not args.cpu: 
                    mask1 = mask1.cuda()
                    mask2 = mask2.cuda()
                if metric.IoU(mask1, mask2) < 0.5:
                    pseudo_mask = pseudo_mask * -1

                # construct binary pseudo-masks
                pseudo_mask[pseudo_mask < 0] = 0
                pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
                pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

                # create coco-style image info
                if img_name not in image_names:
                    image_info = create_image_info(
                        image_id, "{}/{}".format(img_folder, img_name), (height, width, 3))
                    output["images"].append(image_info)
                    image_names.append(img_name)           

                # create coco-style annotation info
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
                if annotation_info is not None:
                    output["annotations"].append(annotation_info)
                    segmentation_id += 1
            image_id += 1

    # save annotations
    if len(img_folders) == args.num_folder_per_job and args.job_index == 0:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N)
    else:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, start_idx, end_idx)
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    print(f'dumping {json_name}')
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
    experiment.log({
        'num_image': len(output['images']),
        'num_ann': len(output['annotations']),
        'json_name': json_name,
    })
    experiment.finish()
