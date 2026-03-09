# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from typing import  Optional

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from tqdm import tqdm
import numpy as np


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class MultiLabelImageFolder(ImageFolder):
    """
    ImageFolder that returns (image, multi_hot) using a user-provided mapping.

    label_mapping: dict that maps "<class>/<filename>" (slash-separated) to a
    1000-d (or nb_classes-d) array/tensor of float32 multi-labels.
    Example key: "n01440764/n01440764_10026.JPEG"
    """

    def __init__(
            self,
            root: str,
            label_mapping_path,
            scheme,
            include_gt,
            threshold=0.5,
            verbose=False,
            transform=None,
            nb_classes: Optional[int] = 1000,
            strict: bool = True,   # True: error on missing key; False: fill zeros
            include_pred_gt=False,
            pred_gt_json=None,

    ):
        super().__init__(root, transform=transform)
        label_map_fn = get_label_mapping_from_json if label_mapping_path.endswith('.json') else get_label_mapping_from_pt
        label_mapping = label_map_fn(
            label_mapping_path, scheme, include_gt, threshold, verbose
        )
        if include_pred_gt:
            label_mapping = self.update_pred_gt(label_mapping, pred_gt_json, verbose)

        # Normalize mapping to torch.float32 tensors once for speed.
        self.nb_classes = (
            int(nb_classes) if nb_classes is not None
            else int(next(iter(label_mapping.values())).shape[0])
        )
        self.strict = strict
        self.label_mapping = label_mapping

    def update_pred_gt(self, original_label_map, pred_gt_json, verbose=False):
        with open(pred_gt_json, "r") as f:
            data = json.load(f)
        if verbose:
            print("updating label map with predicted global labels...")
            iterable = tqdm(original_label_map.items(), total=len(original_label_map))
        else:
            iterable = original_label_map.items()
        for k, v in iterable:
            pred_l, pred_p = data[k]['pred'], data[k]['pred_prob']
            original_label_map[k][int(pred_l)] = max(original_label_map[k][int(pred_l)], float(pred_p))
        del data
        return original_label_map

    def _make_key(self, path: str) -> str:
        # Build "<class>/<filename>" regardless of OS path separator
        cls = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        return f"{cls}/{fname}"

    def __getitem__(self, index: int):
        # image from ImageFolder; we ignore its single-label target
        image, _ = super().__getitem__(index)

        # path to the image in the underlying samples list
        path, _ = self.samples[index]
        key = self._make_key(path)

        target = self.label_mapping.get(key)
        if target is None:
            if self.strict:
                raise KeyError(f"Missing multi-label for key '{key}'")
            # fallback: zeros vector
            target = torch.zeros(self.nb_classes, dtype=torch.float32)

        return image, target


def get_label_mapping_from_json(json_path, scheme, include_gt, threshold=0.5, verbose=False):
    if verbose:
        print(f"loading label map from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    label_mapping_from_json = {}
    if verbose:
        print("building label map from JSON...")
        iterable = tqdm(data.items(), total=len(data))
    else:
        iterable = data.items()
    for k, v in iterable:
        if scheme == 'one_hot':
            cur_label = np.zeros((1000,), dtype=np.float32)
            cur_label[int(v['gt'])] = 1.0
        else:
            cur_label = np.array(v['multilabel'], dtype=np.float32)
            if include_gt:
                cur_label[int(v['gt'])]=1.0
            if scheme == 'relabel_thresh':
                cur_label = (cur_label >= threshold).astype(np.float32)
        label_mapping_from_json[k] = cur_label
    del data
    return label_mapping_from_json

def get_label_mapping_from_pt(json_path, scheme, include_gt, threshold=0.5, verbose=False):
    data = torch.load(json_path, weights_only=False)
    label_mapping_from_json = {}
    if verbose:
        print("building label map from JSON...")
        iterable = tqdm(data.items(), total=len(data))
    else:
        iterable = data.items()
    for k, v in iterable:
        if scheme == 'one_hot':
            cur_label = np.zeros((1000,), dtype=np.float32)
            cur_label[int(v['gt'])] = 1.0
        else:
            cur_label = v['multilabel'].numpy().astype(np.float32)
            if include_gt:
                cur_label[int(v['gt'])]=1.0
            if scheme == 'relabel_thresh':
                cur_label = (cur_label >= threshold).astype(np.float32)
        label_mapping_from_json[k] = cur_label
    del data
    return label_mapping_from_json


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'MULTILABEL':
        # For multi‑label datasets we need an annotation file specifying the labels
        if is_train:
            dataset = MultiLabelImageFolder(
                root=os.path.join(args.data_path, 'train'),
                label_mapping_path=args.relabel_json,
                scheme=args.label_scheme,
                include_gt=args.include_gt,
                threshold=args.threshold,
                verbose=args.rank == 0,
                transform=transform,
                nb_classes = 1000,
                strict = True,
                include_pred_gt=args.include_pred_gt,
                pred_gt_json=args.pred_gt_json,
            )
        else:
            # fall back to standard ImageNet dataset definition for validation
            root = os.path.join(args.data_path, 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
