"""Dataset and patch extraction utilities for the public Patch-MIL pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _safe_imread_color(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _safe_imread_gray(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read mask: {path}")
    return img


def _make_distance_map(mask01: np.ndarray) -> np.ndarray:
    binary = (mask01 > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(mask01, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    max_dist = float(dist.max())
    if max_dist > 0:
        dist /= max_dist
    return dist


def _patch_prior(prior_map: np.ndarray, y: int, x: int, patch_size: int) -> float:
    patch = prior_map[y:y + patch_size, x:x + patch_size]
    return float(patch.mean()) if patch.size else 0.0


def _extract_single_scale(mask01, patch_size, stride, coverage_threshold):
    h, w = mask01.shape[:2]
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    kept = []
    for y in ys:
        for x in xs:
            coverage = float(mask01[y:y + patch_size, x:x + patch_size].mean())
            if coverage >= coverage_threshold:
                kept.append((y, x, patch_size, coverage))
    return kept


def _select_patches(kept, max_count, train_mode, rng):
    if len(kept) >= max_count:
        if train_mode:
            idx = rng.choice(len(kept), size=max_count, replace=False)
            return [kept[i] for i in idx]
        return sorted(kept, key=lambda item: -item[3])[:max_count]
    extra = rng.integers(0, len(kept), size=max_count - len(kept))
    return kept + [kept[i] for i in extra]


def _normalise_patches(patches):
    arr = np.asarray(patches, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def extract_lung_patches_multiscale(
    image_rgb, binary_mask, scales, output_size, max_patches_per_scale,
    train_mode, rng, mask_prior_type="distance"
):
    mask01 = (binary_mask > 0).astype(np.float32)
    distance_map = _make_distance_map(mask01) if mask_prior_type == "distance" else None
    h, w = mask01.shape[:2]
    all_images, all_coords, all_scale_ids, all_priors = [], [], [], []

    for scale_idx, cfg in enumerate(scales):
        ps = int(cfg["patch_size"])
        stride = int(cfg["stride"])
        cov_th = float(cfg["coverage_threshold"])
        kept = _extract_single_scale(mask01, ps, stride, cov_th)
        if not kept:
            kept = [(0, 0, ps, 1.0)]
        final = _select_patches(kept, max_patches_per_scale, train_mode, rng)

        for y, x, ps_, _coverage in final:
            crop = image_rgb[y:y + ps_, x:x + ps_]
            if crop.size == 0:
                crop = np.zeros((ps_, ps_, 3), dtype=np.uint8)
            if ps_ != output_size:
                crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            all_images.append(crop)
            all_coords.append((y / h, x / w))
            all_scale_ids.append(scale_idx)
            if mask_prior_type == "distance":
                prior = _patch_prior(distance_map, y, x, ps_)
            else:
                prior = _patch_prior(mask01, y, x, ps_)
            all_priors.append(prior)

    patches = _normalise_patches(np.stack(all_images, axis=0))
    return (
        patches,
        np.ones((len(all_images),), dtype=np.float32),
        np.asarray(all_coords, dtype=np.float32),
        np.asarray(all_scale_ids, dtype=np.int64),
        np.asarray(all_priors, dtype=np.float32),
    )


def extract_lung_patches(
    image_rgb, binary_mask, patch_size, stride, coverage_threshold,
    max_patches, train_mode, rng, mask_prior_type="distance"
):
    h, w = binary_mask.shape[:2]
    mask01 = (binary_mask > 0).astype(np.float32)
    distance_map = _make_distance_map(mask01) if mask_prior_type == "distance" else None
    kept = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            coverage = float(mask01[y:y + patch_size, x:x + patch_size].mean())
            if coverage >= coverage_threshold:
                kept.append((y, x, coverage))
    if not kept:
        kept = [(0, 0, 1.0)]
    final = _select_patches(kept, max_patches, train_mode, rng)

    patches = np.zeros((max_patches, patch_size, patch_size, 3), dtype=np.float32)
    coords = np.zeros((max_patches, 2), dtype=np.float32)
    priors = np.zeros((max_patches,), dtype=np.float32)
    for i, (y, x, _coverage) in enumerate(final):
        crop = image_rgb[y:y + patch_size, x:x + patch_size]
        patches[i] = _normalise_patches(crop)
        coords[i] = (y / float(h), x / float(w))
        if mask_prior_type == "distance":
            priors[i] = _patch_prior(distance_map, y, x, patch_size)
        else:
            priors[i] = _patch_prior(mask01, y, x, patch_size)

    return patches, np.ones((max_patches,), dtype=np.float32), coords, np.zeros((max_patches,), dtype=np.int64), priors


class PatchMILDataset(Dataset):
    """Load preprocessed CXR images and return lung-aware bags of patches."""

    def __init__(
        self, root_dir, split, canvas_size=384, patch_size=64, stride=32,
        coverage_threshold=0.3, max_patches=100, classes=None, extensions=None,
        augment=False, seed=42, scales=None, multiscale_output_size=64,
        max_patches_per_scale=50, mask_prior_type="distance"
    ):
        self.root_dir = Path(root_dir) / split
        self.canvas_size = int(canvas_size)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.coverage_threshold = float(coverage_threshold)
        self.max_patches = int(max_patches)
        self.augment = augment
        self.classes = classes or ["NORMAL", "PNEUMONIA"]
        self.extensions = {str(x).lower() for x in (extensions or [".png", ".jpg", ".jpeg"])}
        self.rng = np.random.default_rng(seed)
        self.scales = scales
        self.multiscale_output_size = int(multiscale_output_size)
        self.max_patches_per_scale = int(max_patches_per_scale)
        self.mask_prior_type = str(mask_prior_type)
        self.samples = []

        for label, cls in enumerate(self.classes):
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.extensions and "_fullmask" not in p.stem:
                    fullmask = cls_dir / f"{p.stem}_fullmask.png"
                    if fullmask.exists():
                        self.samples.append((p, fullmask, label))

        if not self.samples:
            raise RuntimeError(
                f"No processed samples found in {self.root_dir}. "
                "Expected image + *_fullmask.png."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = _safe_imread_color(str(img_path))
        mask = _safe_imread_gray(str(mask_path))
        image = cv2.resize(image, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)

        if self.augment and self.rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        if self.scales:
            patches, validity, coords, scale_ids, mask_prior = extract_lung_patches_multiscale(
                image, mask, self.scales, self.multiscale_output_size,
                self.max_patches_per_scale, self.augment, self.rng,
                self.mask_prior_type,
            )
        else:
            patches, validity, coords, scale_ids, mask_prior = extract_lung_patches(
                image, mask, self.patch_size, self.stride,
                self.coverage_threshold, self.max_patches, self.augment,
                self.rng, self.mask_prior_type,
            )

        return (
            torch.from_numpy(patches).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(validity),
            torch.from_numpy(coords),
            torch.from_numpy(scale_ids),
            torch.from_numpy(mask_prior),
            label,
        )
