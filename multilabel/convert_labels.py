import torch
from typing import Dict, Any, Optional

def load_from_tsv_sparse(
    tsv_path: str,
    prob_dtype: torch.dtype = torch.float32
) -> Dict[str, Dict[str, Any]]:
    """
    Load labels from TSV file into a sparse-style dict:

    {
        filename: {
            'idx': tensor(K, dtype=int64),
            'p': tensor(K, dtype=prob_dtype),
            'gt': int
        },
        ...
    }
    """
    data = {}

    with open(tsv_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Unexpected line format: {line}")

            filename, gt_str, idx_str, prob_str = parts
            gt = int(gt_str)

            if idx_str.strip() == "":
                # no multilabel entries
                idx = torch.empty(0, dtype=torch.int64)
                probs = torch.empty(0, dtype=prob_dtype)
            else:
                idx_list = [int(x) for x in idx_str.split(",") if x]
                prob_list = [float(x) for x in prob_str.split(",") if x]
                # safety check
                if len(idx_list) != len(prob_list):
                    raise ValueError(f"Length mismatch for {filename}: idx vs prob")

                idx = torch.tensor(idx_list, dtype=torch.int64)
                probs = torch.tensor(prob_list, dtype=prob_dtype)

            data[filename] = {
                "idx": idx,
                "p": probs,
                "gt": gt,
            }

    return data
    
def sparse_to_dense_label_map(
    sparse_data: Dict[str, Dict[str, Any]],
    num_classes: int,
    prob_key: str = "multilabel",
    gt_key: str = "gt",
    prob_dtype: torch.dtype = torch.float32,
):
    """
    Convert sparse-style dict into the original dense style:

    {
        filename: {
            prob_key: tensor(num_classes,),
            gt_key: int
        }
    }
    """
    label_map = {}

    for filename, entry in sparse_data.items():
        idx = entry["idx"].long()
        probs = entry["p"].to(prob_dtype)

        vec = torch.zeros(num_classes, dtype=prob_dtype)
        if idx.numel() > 0:
            vec[idx] = probs

        label_map[filename] = {
            prob_key: vec,
            gt_key: entry["gt"],
        }

    return label_map

