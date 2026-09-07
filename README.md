# Radiology-AI-Pneumonia-PatchMIL

Research-oriented Patch-based Multiple Instance Learning (Patch-MIL) pipeline for pneumonia classification from chest X-ray images.

## Scope

The public repository keeps the **Patch-MIL path only**. Earlier alternative classifier architectures are intentionally excluded.

## Architecture

```text
Chest X-ray
   ↓
Lung preprocessing / lung mask
   ↓
Lung-aware overlapping patches
   ↓
TorchXRayVision DenseNet121 feature encoder
   ↓
Patch embeddings
   ↓
Context Transformer
   ↓
Attention pooling
   ↓
Bag-level pneumonia classifier
```

The cached training path can also use instance-level supervision from available RSNA bounding-box annotations.

## Repository layout

```text
configs/config.yaml
src/
  models/patch_mil.py
  models/lung_segmenter.py
  losses/asymmetric_focal_loss.py
  patch_mil_data_loader.py
  cached_patch_data_loader.py
  train_patch_mil.py
  train_patch_mil_cached.py
  train.py
  scripts/preprocess_segment_dataset.py
  scripts/precompute_patch_cache.py
  utils/config_loader.py
```

## Data

Do **not** commit patient images, DICOM files, private annotations, preprocessing caches, or checkpoints.

Expected processed structure:

```text
data_processed/
  train/NORMAL/
  train/PNEUMONIA/
  val/NORMAL/
  val/PNEUMONIA/
  test/NORMAL/
  test/PNEUMONIA/
```

For the raw Patch-MIL loader, each image is expected to have a corresponding `*_fullmask.png` lung mask.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Then:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml` for local paths and experiment settings. Never put machine-specific paths, credentials, or private dataset locations into the public repository.

## Preprocessing

```bash
python -m src.scripts.preprocess_segment_dataset
```

## Raw Patch-MIL training

```bash
python -m src.train_patch_mil
```

## Cached Patch-MIL training

First generate cached embeddings:

```bash
python -m src.scripts.precompute_patch_cache
```

Then train from the cached embeddings:

```bash
python -m src.train_patch_mil_cached
```

## Evaluation

The training scripts select the best checkpoint by validation AUC, calibrate the decision threshold on validation data for the requested sensitivity target, and then apply that fixed threshold to the held-out test set.

## Limitations

This is a research prototype. It has not been clinically validated and must not be presented as a diagnostic medical device. Shortcut learning, dataset shift, calibration, and external validation remain active research questions.

## License

See `LICENSE` and `docs/DATA_AND_LICENSE.md`. A final open-source license should only be published after checking the licensing terms of all included and referenced third-party components and pretrained weights.
