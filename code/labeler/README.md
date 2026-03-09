# Labeler - Classification Head Training

## Scripts

### `mask_level_relabel.py`
Derives confidence scores corresponding to the original ImageNet label based on object proposal masks and ReLabel's label map. Used for filtering proposals for labeler training.

### `train_labeler.py`
Trains a classification head on top of a frozen backbone, based on localized feature areas corresponding to original labels.

### `relabel_trainset.py`
Performs inference on all object proposals and saves results for later multi-label training.

