# Multilabel ImageNet-1K Annotations (Compressed Format)

We provide **multilabel annotations for ImageNet-1K** in a **compact text format** intended for review.

To minimize storage size, probabilities are quantized (rounded to 4 decimal places), and the full object proposals (≈16.3 GB) are omitted, which will be released together upon paper acceptance.

------

## **File Format: `multilabel_annotations.tsv`**

Each line corresponds to a single image and follows the layout:

```
filename<TAB>gt_index<TAB>idx_list<TAB>prob_list
```

### **Example**

```
n01440764/n01440764_10042.JPEG    0    0,391    0.2654,1.0000
```

### **Field definitions**

- **`filename`** — The image path relative to the ImageNet root directory.
- **`gt_index`** — The original ImageNet-1K ground-truth class index (0–999).
- **`idx_list`** — A comma-separated list of class indices assigned non-zero multilabel probability.
- **`prob_list`** — The corresponding probabilities, rounded to **4** decimal places.

------

## **Reconstruction Code**

We provide a reference Python script, **`convert_labels.py`**, which can:

- Load the `.tsv` annotations.
- Reconstruct **sparse** per-image label tensors (`idx`, `prob`, `gt`).
- Optionally recover **dense** 1000-dimensional label vectors in the original format.