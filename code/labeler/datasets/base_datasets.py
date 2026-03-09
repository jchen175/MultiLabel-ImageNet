import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import numpy as np
from PIL import Image
import torch
from pycocotools.mask import decode as coco_decode
import json
import cv2
import math
from typing import Any, Tuple
from collections import defaultdict


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225])

def imagenet_transform_cpu(size, mean=IMNET_MEAN, std=IMNET_STD, split='val', center_crop=False, rand_aug=False):
    """
    Get ImageNet transformation for CPU processing
    :param size: resize size for the image
    :param mean: ImageNet mean for normalization
    :param std: ImageNet std for normalization
    :param split: 'train'/'val' to determine the type of transformation
    :param center_crop: whether to apply center_crop on the validation set
    :param rand_aug: whether to apply random aug n train set
    :return:
    """
    assert split in ['train', 'val'], f"Invalid split: {split}. Must be 'train' or 'val'."
    trans_list = []
    if split == 'val':
        if not center_crop:
            trans_list.append(transforms.Resize((size, size)))
        else:
            resize_ = int(256/224*size)
            trans_list.extend([transforms.Resize(resize_), transforms.CenterCrop(size)])
    if split == 'train':
        trans_list.append(transforms.RandomResizedCrop((size, size)))
        trans_list.append(transforms.RandomHorizontalFlip())
        if rand_aug:
            trans_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    trans_list.extend([transforms.ToTensor(),
                       transforms.Normalize(mean=mean, std=std)])
    transform = transforms.Compose(trans_list)
    return transform

class ReaLValSet(Dataset):
    """
    add order information to calculate the ReaL label acc
    """
    def __init__(self, root, val_seg_json=None, transform=None):
        self.base_dataset = ImageFolder(root=root, transform=transform)


    def __len__(self):
        return len(self.base_dataset)

    def get_id(self, filename):
        basename = os.path.basename(filename)
        index = int(basename.split('_')[2])
        return index

    def __getitem__(self, index):
        item = self.base_dataset[index]
        out = {
            'image': item[0],
            'label': item[1],
            'index': self.get_id(self.base_dataset.imgs[index][0]),
            'name': os.path.basename(self.base_dataset.imgs[index][0])
        }
        return out

class ImageNetHelper:
    """
    A helper class to manage ImageNet dataset operations such as loading images, class mappings, vague class name searches, etc.
    """
    def __init__(self, root_dir="<DATA_ROOT>/dataset/imagenet/", class_file="LOC_synset_mapping.txt"):
        self.root_dir = root_dir
        self.class_file = class_file
        self.class_mapping = None
        self.class_id2idx = None
        self.class_list = []
        self.class_list_lower = []
        self.load_class_mapping()

    def load_class_mapping(self):
        with open(os.path.join(self.root_dir, self.class_file), "r") as f:
            lines = f.readlines()
        class_mapping = {}
        class_id2idx = {}
        for idx, line in enumerate(lines):
            split_line = line.strip().split(" ")
            class_id = split_line[0]
            class_name = ' '.join(split_line[1:])
            class_mapping[class_id] = class_name
            class_id2idx[class_id] = idx
            self.class_list.append(class_name)
            self.class_list_lower.append(class_name.lower())
        self.class_mapping = class_mapping
        self.class_id2idx = class_id2idx
        self.class_name2id = {v: k for k, v in class_mapping.items()}


    def get_class_name(self, filename):
        if 'val' in filename:
            class_id = filename.split('_')[-1][:-5]
        else:
            class_id = os.path.basename(filename).split("_")[0]
        return self.class_mapping[class_id]

    def get_class_index(self, filename):
        if '_val_' in filename:
            class_id = filename.split('_')[-1][:-5]
        else:
            class_id = os.path.basename(filename).split("_")[0]
        return self.class_id2idx[class_id]

    def get_class_name_by_index(self, index):
        return self.class_list[index]


    def open_image(self, filename):
        image_path = self.get_image_path(filename)
        with Image.open(image_path) as img:
            return img.convert("RGB").copy()  # copy ensures data is detached from file


    def get_image_path(self, filename):
        if 'val' in filename:
            class_id = filename.split('_')[-1][:-5]
            filename = os.path.basename(filename).split(".")[0] + ".JPEG"
            val_dir = os.path.join(self.root_dir, "val", class_id, filename)
            return val_dir
        else:
            class_id = os.path.basename(filename).split("_")[0]
            filename = os.path.basename(filename).split(".")[0] + ".JPEG"
            train_dir = os.path.join(self.root_dir, "train", class_id, filename)
            if os.path.exists(train_dir):
                return train_dir
            else:
                val_dir = os.path.join(self.root_dir, "val", class_id, filename)
                return val_dir

    def get_random_image(self, split = "train"):
        cat = random.choice(os.listdir(os.path.join(self.root_dir, split)))
        filename = random.choice(os.listdir(os.path.join(self.root_dir, split, cat)))
        return {
            'image': self.open_image(os.path.join(cat, filename)),
            'class': self.get_class_name(os.path.join(cat, filename)),
            'filename': filename
        }

    def get_random_image_from_class_name(self, class_name):
        cat = self.class_name2id[class_name]
        filename = random.choice(os.listdir(os.path.join(self.root_dir, "train", cat)))
        return {
            'image': self.open_image(os.path.join(cat, filename)),
            'class': self.get_class_name(os.path.join(cat, filename)),
            'filename': filename
        }

    def get_random_image_from_class_index(self, class_index):
        class_name = self.class_list[class_index]
        cat = self.class_name2id[class_name]
        filename = random.choice(os.listdir(os.path.join(self.root_dir, "train", cat)))
        return {
            'image': self.open_image(os.path.join(cat, filename)),
            'class': self.get_class_name(os.path.join(cat, filename)),
            'filename': filename
        }

    def get_random_image_from_class_id(self, class_id):
        filename = random.choice(os.listdir(os.path.join(self.root_dir, "train", class_id)))
        return {
            'image': self.open_image(os.path.join(class_id, filename)),
            'class': self.get_class_name(os.path.join(class_id, filename)),
            'filename': filename
        }

    def get_random_image_from_vague_class_name(self, vague_class_name):
        found_class_name = []
        for class_name in self.class_list:
            if vague_class_name in class_name:
                found_class_name.append(class_name)
        if len(found_class_name) == 0:
            print(f"No class name found for {vague_class_name}")
            return None
        elif len(found_class_name) == 1:
            return self.get_random_image_from_class_name(found_class_name[0]),
        else:
            print(f"Found {len(found_class_name)} classes for {vague_class_name}; please select one of the following:")
            for i, class_name in enumerate(found_class_name):
                print(f"{i}: {class_name}")
            selected_index = int(input("Please select the index of the class you want to use: "))
            return self.get_random_image_from_class_name(found_class_name[selected_index])

    def get_vague_class_index(self, vague_class_name):
        found_class_name = []
        for class_name in self.class_list_lower:
            if vague_class_name.lower() in class_name:
                found_class_name.append(class_name)
        if len(found_class_name) == 0:
            print(f"No class name found for {vague_class_name}")
            return None
        elif len(found_class_name) == 1:
            print(f"Full class name for {vague_class_name} is {found_class_name[0]}")
            return self.class_name2id[found_class_name[0]]
        else:
            print(
                f"Found {len(found_class_name)} classes for {vague_class_name}; please select one of the following:")
            for i, class_name in enumerate(found_class_name):
                print(f"{i}: {class_name}")
            selected_index = int(input("Please select the index of the class you want to use: "))
            return self.class_name2id[found_class_name[selected_index]]

def get_mask_augmentation(
    crop_size = 224,
    *,
    scale_range: tuple[float, float] = (0.08, 1.0),
    ratio_range: tuple[float, float] = (3/4, 4/3),
    use_color_jitter: bool = True,
    ensure_nonempty_mask: bool = True,
):
    # lazy imports
    import numpy as np
    import albumentations as A
    from albumentations.core.transforms_interface import DualTransform
    from albumentations.pytorch import ToTensorV2

    """
    Returns an Albumentations Compose that can be called with
        out = aug(image=img, mask=mask)   # mask may be None
    and yields a dict with keys "image" (Tensor C×H×W) and "mask" (Tensor H×W).
    """

    # ---------- Safe replacement transform ----------
    class SafeCropNonEmptyMaskIfExists(DualTransform):
        def __init__(self, height, width, p=1.0):
            super().__init__(p=p)
            self.height = int(height)
            self.width  = int(width)

        @property
        def targets_as_params(self):
            return ["mask"]

        def get_params_dependent_on_data(self, params, data):
            img  = data.get("image", None)
            mask = data.get("mask", None)

            # If image is missing, do nothing.
            if img is None:
                return {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}

            H, W = img.shape[:2]

            # If target size equals current size, no-op
            if H == self.height and W == self.width:
                return {"x_min": 0, "y_min": 0, "x_max": W, "y_max": H}

            # Build a 2D boolean mask if possible
            coords = None
            if mask is not None:
                m = np.asarray(mask)
                if m.ndim == 3 and m.shape[2] == 1:
                    m = m[..., 0]
                if m.ndim == 3:
                    m = (m != 0).any(axis=2)
                else:
                    m = (m != 0)
                if m.size and m.any():
                    coords = np.argwhere(m)

            if coords is None or coords.size == 0:
                # Fallback: random crop anywhere within bounds
                y1 = np.random.randint(0, max(1, H - self.height + 1))
                x1 = np.random.randint(0, max(1, W - self.width  + 1))
            else:
                # Choose a positive pixel index via NumPy (no truthiness ambiguity)
                y, x = coords[np.random.randint(len(coords))]
                y1 = int(np.clip(y - self.height // 2, 0, max(0, H - self.height)))
                x1 = int(np.clip(x - self.width  // 2, 0, max(0, W - self.width )))

            return {"x_min": x1, "y_min": y1, "x_max": x1 + self.width, "y_max": y1 + self.height}

        def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, **params):
            return img[y_min:y_max, x_min:x_max, ...]

        def apply_to_mask(self, mask, x_min=0, y_min=0, x_max=0, y_max=0, **params):
            if mask is None:
                return None
            return mask[y_min:y_max, x_min:x_max, ...]

        def get_transform_init_args_names(self):
            return ("height", "width")

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)

    h, w = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    # ---- geometric & pixel transforms ----------------------------
    geo = [
        A.RandomResizedCrop(
            height=h, width=w,
            scale=scale_range,
            ratio=ratio_range,
            interpolation=1,          # bilinear for image; mask handled as NN internally
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=20,
            border_mode=0, value=0, mask_value=0,
            p=0.7
        ),
    ]

    pixel = []
    if use_color_jitter:
        pixel.extend([
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        ])

    # ---- choose crop operator depending on mask requirement -------
    if ensure_nonempty_mask:
        crop_op = SafeCropNonEmptyMaskIfExists(height=h, width=w, p=1.0)
    else:
        crop_op = A.CenterCrop(height=h, width=w, p=1.0)

    # ---- compose final pipeline -----------------------------------
    aug = A.Compose(
        geo + pixel + [
            crop_op,
            A.Normalize(_IMAGENET_MEAN, _IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),  # image -> (C,H,W); mask -> (H,W)
        ],
        additional_targets={'mask': 'mask'}  # keeps image & mask in sync
    )
    return aug

class MCutDataset(Dataset):
    """
    return each mask annotation and its corresponding original image for an annotation file;
    """
    def __init__(self, maskcut_annotation_file, transform=None, imagenet_helper=None, mask_size=32):
        if imagenet_helper is None:
            self.helper = ImageNetHelper()
        else:
            self.helper = imagenet_helper
        self.mcut_annotation = self._load_annotation(maskcut_annotation_file)
        self.masks = self.mcut_annotation['annotations']
        self.images = self.mcut_annotation['images']
        self.mask_size = mask_size

        self.transform = transform

    def _resize_mask(self, input_mask):
        resized_mask = cv2.resize(input_mask, (self.mask_size, self.mask_size), interpolation=cv2.INTER_LINEAR)
        return resized_mask


    def _load_annotation(self, file_path):
        with open(file_path, 'r') as f:
            annotation = json.load(f)
        return annotation

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        cur_annotation = self.masks[idx]
        mask_id = cur_annotation['id']
        image_index = cur_annotation['image_id']-1
        image_file = self.images[image_index]['file_name']
        image = self.helper.open_image(image_file)
        image = self.transform(image)
        mask = coco_decode(cur_annotation['segmentation'])
        mask = self._resize_mask(mask)
        mask = torch.from_numpy(mask)
        return {
            'image': image,
            'mask': mask,
            'mask_id': mask_id,
        }


class RelabelMaskcutDataset(Dataset):
    # merge masks on an image base
    def __init__(self, seg_annotation, mask_label_dict, threshold=0.9, force_gt_label=False, transform=None, helper=None, mask_size=32):
        from albumentations.pytorch import ToTensorV2
        self.seg_annotation = seg_annotation
        self.mask_label_dict = mask_label_dict
        self.helper = helper
        self.mask_size = mask_size
        self.threshold = threshold
        self.force_gt_label = force_gt_label
        self.transform = transform

        self._get_confident_mask()
        self._get_image_mask_mapping()
        self._merge_mask_by_label()
        self.to_tensor = ToTensorV2()

    def _get_confident_mask(self):
        self.confident_mask = {}
        for k, v in self.mask_label_dict.items():
            if v['prob'][0] >= self.threshold:
                self.confident_mask[k] = v['label'][0]

    def _get_image_mask_mapping(self):
        self.mask_id2image_id = {}
        for ann in self.seg_annotation['annotations']:
            self.mask_id2image_id[ann['id']] = ann['image_id']
        self.image_id2mask_id = defaultdict(list)
        for k, v in self.mask_id2image_id.items():
            self.image_id2mask_id[v].append(k)

    def _merge_mask_by_label(self):
        self.image_masklist_label = []
        print(f"Merge mask by label: {self.threshold}")
        for img in self.seg_annotation['images']:
            gt_label = self.helper.get_class_index(img['file_name'])
            gt_hit = False
            if img['id'] in self.image_id2mask_id:
                cur_masks = []
                cur_labels = []
                for mask_id in self.image_id2mask_id[img['id']]:
                    if mask_id in self.confident_mask:
                        cur_masks.append(mask_id)
                        cur_labels.append(self.confident_mask[mask_id])
                unique_values = np.unique(cur_labels)
                for v in unique_values:
                    if v == gt_label:
                        gt_hit = True
                    local_masks = []
                    for l, m in zip(cur_labels, cur_masks):
                        if l == v:
                            local_masks.append(m)
                    self.image_masklist_label.append([img['id'], local_masks, v])
            if not gt_hit and self.force_gt_label:
                self.image_masklist_label.append([img['id'], [], gt_label])


    def __len__(self):
        return len(self.image_masklist_label)

    def _load_image_by_id(self, img_id):
        img = self.seg_annotation['images'][img_id-1]
        assert img_id == img['id']
        img = self.helper.open_image(img['file_name'])
        return img

    def _resize_mask(self, mask, size=None, mode: str = "bilinear"):
        """
        Resize a mask tensor to (size, size) using PyTorch ops.

        Args:
            mask: torch.Tensor of shape (H, W) or (1, H, W) or (B, 1, H, W).
            size: int, target side length. Defaults to self.mask_size.
            mode: 'bilinear' (default) for soft/probability masks; use 'nearest' for label masks.

        Returns:
            torch.Tensor of shape (H, W), dtype=float32.
        """
        if size is None:
            size = self.mask_size

        # Ensure tensor
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        device = mask.device

        # Cast to float for interpolations like bilinear/bicubic
        t = mask.to(torch.float32)

        # Normalize shape to (B, C, H, W)
        if t.ndim == 2:  # (H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            # Accept (1, H, W) or (C, H, W). If it's (H, W, C) from some path, permute.
            if t.shape[-1] in (1, 3) and t.shape[0] != 1:
                # (H, W, C) -> (C, H, W)
                t = t.permute(2, 0, 1).unsqueeze(0)
            else:
                t = t.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        elif t.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported mask shape {tuple(t.shape)}")

        # Interpolate
        align = False if mode in ("bilinear", "bicubic", "trilinear") else None
        out = F.interpolate(t, size=(size, size), mode=mode, align_corners=align)

        # Squeeze back to (H, W)
        out = out.squeeze(0)  # (C, H, W) or (H, W)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        elif out.ndim == 3 and out.shape[0] > 1:
            # If you ever pass multi-channel masks, pick channel 0 or adapt as needed.
            out = out[0]

        return out.to(device)  # float32

    def _load_mask_by_id(self, mask_id):
        mask = self.seg_annotation['annotations'][mask_id-1]
        assert mask['id'] == mask_id
        mask = coco_decode(mask['segmentation'])
        return mask

    def _load_mask_by_id_list(self, mask_ids):
        masks = []
        for mask_id in mask_ids:
            mask = self._load_mask_by_id(mask_id)
            masks.append(mask)
        final_mask = np.sum(masks, axis=0)
        final_mask = np.clip(final_mask, 0, 1)
        return final_mask


    def __getitem__(self, idx):
        image_id, mask_id_list, label = self.image_masklist_label[idx]
        image = self._load_image_by_id(image_id)
        h, w= image.size[:2]
        image = np.array(image)
        if len(mask_id_list) == 0:

            mask = np.ones((w,h))
        else:
            mask = self._load_mask_by_id_list(mask_id_list)

        out = self.transform(image=image, mask=mask)
        mask = self._resize_mask(out['mask'], self.mask_size)


        out = {
            # 'image': self.to_tensor(image=out['image'])['image'],
            'image': out['image'],
            'mask': mask,
            'label': label,
        }
        return out


def split_dataset_indices(total_length, total_jobs, job_index):
    """Split dataset indices for parallel processing"""
    if job_index >= total_jobs:
        raise ValueError(f"job_index {job_index} must be less than total_jobs {total_jobs}")

    # Calculate the size of each chunk
    chunk_size = math.ceil(total_length / total_jobs)

    # Calculate start and end indices for this job
    start_idx = job_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_length)

    return start_idx, end_idx


class SubsetDataset(torch.utils.data.Dataset):
    """Wrapper to create a subset of the original dataset"""
    def __init__(self, dataset, start_idx, end_idx):
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.length = end_idx - start_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of range")
        return self.dataset[self.start_idx + idx]


class ImageFolderWithFilenames(ImageFolder):
    def __init__(self, root: str, transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Any, int, str]:
        # Get (image, target) using ImageFolder's logic (handles loading + transforms)
        image, target = super().__getitem__(index)

        # Robustly get the file path from ImageFolder (works across torchvision versions)
        # ImageFolder stores file list in either `samples` or `imgs`
        path_list = getattr(self, "samples", getattr(self, "imgs", None))
        if path_list is None:
            # Fallback (shouldn't happen on normal ImageFolder)
            raise RuntimeError("Cannot find `samples`/`imgs` on ImageFolder. torchvision version mismatch?")
        path = path_list[index][0]

        filename = '/'.join(path.split('/')[-2:])  # relative path with class subdir
        return image, target, filename