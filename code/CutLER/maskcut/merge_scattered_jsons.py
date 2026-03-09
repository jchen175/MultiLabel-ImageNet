#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all COCO‐style JSON annotation files in a directory into one,
reassigning image and annotation IDs to avoid collisions.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

def merge_coco_jsons(
    json_paths: List[Path],
) -> Dict[str, Any]:
    """
    Merge multiple COCO‐style JSONs into a single dict.
    Returns a dict with keys: 'images', 'annotations', plus any
    other keys (e.g. 'categories') taken from the first JSON that has them.
    """
    merged = {}
    merged["images"] = []
    merged["annotations"] = []

    next_img_id = 1
    next_ann_id = 1

    # carry-over of other keys (like categories) from the first file that has them
    carried_keys = {}

    for p in tqdm(json_paths):
        data = json.loads(p.read_text())
        imgs = data.get("images")
        anns = data.get("annotations")

        if imgs is None or anns is None:
            logging.warning(f"Skipping {p.name}: missing 'images' or 'annotations'")
            continue

        # record other keys once
        if not carried_keys:
            for k, v in data.items():
                if k not in ("images", "annotations"):
                    carried_keys[k] = v

        # map old image IDs → new image IDs
        id_map: Dict[int,int] = {}
        for img in imgs:
            old_id = img["id"]
            img["id"] = next_img_id
            merged["images"].append(img)
            id_map[old_id] = next_img_id
            next_img_id += 1

        # remap each annotation
        for ann in anns:
            old_img_id = ann["image_id"]
            ann["image_id"] = id_map.get(old_img_id)
            ann["id"] = next_ann_id
            # ensure iscrowd
            ann["iscrowd"] = ann.get("iscrowd", 0)
            merged["annotations"].append(ann)
            next_ann_id += 1

    # attach any carried keys
    merged.update(carried_keys)
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Merge all COCO‐style JSONs in a directory into one."
    )
    parser.add_argument(
        "--base-dir", "-b",
        type=Path,
        required=True,
        help="Directory containing .json annotation files."
    )
    parser.add_argument(
        "--save-path", "-s",
        type=Path,
        required=True,
        help="Where to write the merged JSON."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    json_paths = sorted(args.base_dir.glob("*.json"))
    if not json_paths:
        logging.error(f"No .json files found in {args.base_dir}")
        return

    logging.info(f"Found {len(json_paths)} files. Merging…")
    merged = merge_coco_jsons(json_paths)

    # final sanity check
    num_images = len(merged.get("images", []))
    num_anns   = len(merged.get("annotations", []))
    logging.info(f"Merged → {num_images} images, {num_anns} annotations")

    # write out
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(merged, f, indent=2)
    logging.info(f"Saved merged JSON to {args.save_path}")

if __name__ == "__main__":
    main()
