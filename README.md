# Radiology-AI-Pneumonia-PatchMIL

Research-oriented **Patch-based Multiple Instance Learning (Patch-MIL)** pipeline for pneumonia classification from chest X-ray (CXR) images.

> **Research software:** This repository is not a clinically validated diagnostic device.

## What is included

This public repository intentionally contains **one classifier architecture: Patch-MIL**.

```text
Chest X-ray
    │
    ▼
Black-border removal + lung segmentation
    │
    ▼
Lung-aware overlapping patches
    │
    ▼
TorchXRayVision DenseNet121 encoder
    │
    ▼
Patch embeddings
    │
    ▼
Patch-context Transformer
    │
    ▼
Lung-prior attention pooling
    │
    ▼
Bag-level pneumonia prediction
```

The project supports two execution paths:

- **Raw Patch-MIL:** extracts patches and runs the CXR encoder during training.
- **Cached Patch-MIL:** precomputes encoder embeddings once and trains the MIL/context/attention head from `.npz` files.

The cached path also supports optional RSNA bounding-box supervision when a **local** annotation CSV is supplied. No patient data, DICOM files, caches, checkpoints, or private annotations belong in this repository.

## Repository structure

```text
Radiology-AI-Pneumonia-PatchMIL/
├── configs/
│   └── config.yaml
├── docs/
│   └── DATA_AND_LICENSE.md
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── train_patch_mil.py
│   ├── train_patch_mil_cached.py
│   ├── patch_mil_data_loader.py
│   ├── cached_patch_data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patch_mil.py
│   │   └── lung_segmenter.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── asymmetric_focal_loss.py
│   ├── scripts/
│   │   ├── preprocess_segment_dataset.py
│   │   └── precompute_patch_cache.py
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py
├── tests/
│   ├── test_config.py
│   └── test_patch_mil.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All project paths and experiment parameters live in:

```text
configs/config.yaml
```

The public configuration uses **relative paths**. Do not place local machine paths, credentials, patient information, private annotation paths, or downloaded model weights in the public file.

For a private/local override:

```bash
set RADIOLOGY_CONFIG_PATH=C:\path\to\your\config.yaml
```

or on Linux/macOS:

```bash
export RADIOLOGY_CONFIG_PATH=/path/to/your/config.yaml
```

## Data layout

The preprocessing/training pipeline expects the original image data locally:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

After preprocessing, the expected structure is:

```text
data_processed/
├── train/NORMAL/
├── train/PNEUMONIA/
├── val/NORMAL/
├── val/PNEUMONIA/
├── test/NORMAL/
└── test/PNEUMONIA/
```

Each processed image has a corresponding lung mask named:

```text
<image_stem>_fullmask.png
```

### Important

Do **not** upload these directories to GitHub:

- `data/`
- `data_processed/`
- `external_test/`
- `patch_cache/`
- `weights/`
- DICOM files
- patient images
- annotation CSVs containing restricted data

They are already covered by `.gitignore`.

## Preprocessing

Run:

```bash
python -m src.scripts.preprocess_segment_dataset
```

The preprocessing stage removes black borders, generates a lung mask, and writes the processed image plus `*_fullmask.png`.

## Train Patch-MIL from raw patches

```bash
python -m src.train_patch_mil
```

The current public configuration uses:

- TorchXRayVision DenseNet121 (`densenet121-res224-all`)
- 64×64 patches
- stride 32
- lung coverage threshold 0.3
- up to 100 patches per image
- 2 Transformer context layers
- 8 attention heads
- distance-based lung prior
- partial fine-tuning of the final DenseNet block after the freeze period
- asymmetric focal loss with label smoothing

## Cached Patch-MIL

Precompute embeddings:

```bash
python -m src.scripts.precompute_patch_cache
```

Then train:

```bash
python -m src.train_patch_mil_cached
```

Cached files are stored under `patch_cache/` and are intentionally ignored by Git.

## Training outputs

Checkpoints and metrics are written to:

```text
weights/patch_mil/raw/
weights/patch_mil/cached/
```

Typical artifacts include:

```text
best_model.pt
model_arch.json
calibration.json
test_metrics.json
```

These are also ignored by Git.

## Evaluation protocol

The training scripts:

1. select the best checkpoint using validation AUC;
2. calibrate a classification threshold on the validation set to meet the configured sensitivity target;
3. apply that fixed threshold to the held-out test set;
4. report AUC, sensitivity, specificity, and the selected threshold.

This threshold is **not** tuned on the test set.

## Testing the codebase

Run:

```bash
python -m pytest -q
```

The tests cover configuration loading and core Patch-MIL attention/token-path behavior. They do not constitute clinical validation.

## Reproducibility

The main experiment seed is configured in `configs/config.yaml`:

```yaml
project:
  seed: 42
```

For meaningful scientific comparison, record the configuration, dataset version/split, preprocessing version, pretrained-weight version, random seed, and software environment for every experiment.

## Commercial use

The repository's original code is released under the MIT License. **That does not automatically grant commercial rights to third-party datasets, pretrained weights, or external services.** Review their individual terms before using this project in a commercial product.

In particular, the RSNA Pneumonia Detection Challenge data has its own terms and attribution requirements, and TorchXRayVision notes that licensing can vary by subpackage/model. See `docs/DATA_AND_LICENSE.md` before any commercial deployment.

## Citation

If this repository contributes to a publication, benchmark, or technical report, please cite the repository and the relevant datasets/models used in the experiment.

## Disclaimer

This software is provided for research and engineering purposes. It has not been clinically validated, and model performance can change substantially under dataset shift, different acquisition protocols, different patient populations, or different hospital environments.
