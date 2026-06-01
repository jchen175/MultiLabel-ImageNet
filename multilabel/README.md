# Multilabel ImageNet-1K Annotations

We provide **multilabel annotations for ImageNet-1K** with two access paths:

- Complete Hugging Face release: https://huggingface.co/datasets/k3999/multilabel-imagenet-1k
- Lightweight image-level TSV in this repository: `multilabel_compressed.tsv`

We used the standard [LOC_synset_mapping](https://github.com/formigone/tf-imagenet/blob/master/LOC_synset_mapping.txt), which provides the correspondence between class indices and class names.

The Hugging Face release contains the ImageNet-1K train-split multi-label annotations, selected object masks, per-mask top-5 label probabilities, and export metadata. It does **not** include the original ImageNet images; users need separate access to ImageNet-1K and can join annotations by the relative `filename` field.

The in-repository TSV is kept as a compact image-level annotation file for quick inspection and compatibility. To minimize storage size, probabilities in the TSV are quantized to 4 decimal places and mask-level annotations are stored in the Hugging Face dataset instead.

------

## **Complete Hugging Face Dataset**

The Hugging Face dataset includes:

- `image_labels.parquet`: one row per ImageNet-1K training image, with sparse image-level label indices and probabilities.
- `mask_labels.parquet`: at most one selected mask per `(image, positive label)`, including COCO RLE mask encoding and top-5 per-mask predictions.
- `metadata.json`: source/config metadata and the mask selection rule.

For each positive image-level label, the released mask table keeps the selected mask with the highest probability for that class across the proposal configurations. This matches the aggregation rule used to produce the compressed image-level labels.

Example loading:

```python
from datasets import load_dataset

repo_id = "k3999/multilabel-imagenet-1k"

image_labels = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{repo_id}/image_labels.parquet",
    split="train",
)

mask_labels = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{repo_id}/mask_labels.parquet",
    split="train",
)
```

------

## **Compressed TSV Format: `multilabel_compressed.tsv`**

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

- Load the compressed `.tsv` annotations.
- Reconstruct **sparse** per-image label tensors (`idx`, `prob`, `gt`).
- Optionally recover **dense** 1000-dimensional label vectors in the original format.
