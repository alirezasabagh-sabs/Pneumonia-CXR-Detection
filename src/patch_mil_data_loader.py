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
        raise ValueError(f'Cannot read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _safe_imread_gray(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Cannot read mask: {path}')
    return img

def _extract_single_scale(image_rgb, mask01, patch_size, stride, coverage_threshold):
    h, w = mask01.shape[:2]
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    kept = []
    for y in ys:
        for x in xs:
            coverage = mask01[y:y+patch_size, x:x+patch_size].mean()
            if coverage >= coverage_threshold:
                kept.append((y, x, patch_size, coverage))
    return kept

def extract_lung_patches_multiscale(image_rgb, binary_mask, scales, output_size,
                                    max_patches_per_scale, train_mode, rng):
    mask01 = (binary_mask > 0).astype(np.float32)
    h, w = mask01.shape[:2]
    all_images, all_coords, all_scale_ids = [], [], []
    for scale_idx, cfg in enumerate(scales):
        ps, stride, cov_th = cfg['patch_size'], cfg['stride'], cfg['coverage_threshold']
        kept = _extract_single_scale(image_rgb, mask01, ps, stride, cov_th)
        if not kept:
            kept = [(0, 0, ps, 1.0)]
        if len(kept) >= max_patches_per_scale:
            if train_mode:
                idx = rng.choice(len(kept), size=max_patches_per_scale, replace=False)
                final = [kept[i] for i in idx]
            else:
                final = sorted(kept, key=lambda c: -c[3])[:max_patches_per_scale]
        else:
            extra = rng.integers(0, len(kept), size=max_patches_per_scale-len(kept))
            final = kept + [kept[i] for i in extra]
        for y, x, ps_, _ in final:
            crop = image_rgb[y:y+ps_, x:x+ps_]
            if ps_ != output_size:
                crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            all_images.append(crop)
            all_coords.append((y / h, x / w))
            all_scale_ids.append(scale_idx)
    total = len(all_images)
    patches = np.zeros((total, output_size, output_size, 3), dtype=np.float32)
    coords = np.zeros((total, 2), dtype=np.float32)
    scale_ids = np.zeros((total,), dtype=np.int64)
    for i, (crop, coord, sid) in enumerate(zip(all_images, all_coords, all_scale_ids)):
        norm = crop.astype(np.float32) / 255.0
        norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
        patches[i], coords[i], scale_ids[i] = norm, coord, sid
    validity = np.ones((total,), dtype=np.float32)
    return patches, validity, coords, scale_ids

def extract_lung_patches(image_rgb, binary_mask, patch_size, stride, coverage_threshold,
                         max_patches, train_mode, rng):
    h, w = binary_mask.shape[:2]
    mask01 = (binary_mask > 0).astype(np.float32)
    kept = []
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            coverage = mask01[y:y+patch_size, x:x+patch_size].mean()
            if coverage >= coverage_threshold:
                kept.append((y, x, coverage))
    if not kept:
        kept = [(0, 0, 1.0)]
    if len(kept) >= max_patches:
        if train_mode:
            idx = rng.choice(len(kept), size=max_patches, replace=False)
            final = [kept[i] for i in idx]
        else:
            final = sorted(kept, key=lambda c: -c[2])[:max_patches]
    else:
        extra = rng.integers(0, len(kept), size=max_patches-len(kept))
        final = kept + [kept[i] for i in extra]
    patches = np.zeros((max_patches, patch_size, patch_size, 3), dtype=np.float32)
    coords = np.zeros((max_patches, 2), dtype=np.float32)
    scale_ids = np.zeros((max_patches,), dtype=np.int64)
    validity = np.ones((max_patches,), dtype=np.float32)
    for i, (y, x, _) in enumerate(final):
        crop = image_rgb[y:y+patch_size, x:x+patch_size].astype(np.float32) / 255.0
        patches[i] = (crop - IMAGENET_MEAN) / IMAGENET_STD
        coords[i] = (y / float(h), x / float(w))
    return patches, validity, coords, scale_ids, len(kept)

class PatchMILDataset(Dataset):
    def __init__(self, root_dir, split, canvas_size=384, patch_size=64, stride=32,
                 coverage_threshold=0.3, max_patches=100, classes=None, extensions=None,
                 augment=False, seed=42, scales=None, multiscale_output_size=64,
                 max_patches_per_scale=50):
        self.root_dir = Path(root_dir) / split
        self.canvas_size = canvas_size; self.patch_size = patch_size; self.stride = stride
        self.coverage_threshold = coverage_threshold; self.max_patches = max_patches
        self.augment = augment; self.classes = classes or ['NORMAL', 'PNEUMONIA']
        self.extensions = set(extensions or ['.png', '.jpg', '.jpeg'])
        self.rng = np.random.default_rng(seed); self.scales = scales
        self.multiscale_output_size = multiscale_output_size
        self.max_patches_per_scale = max_patches_per_scale
        self.samples = []
        for label, cls in enumerate(self.classes):
            cls_dir = self.root_dir / cls
            if not cls_dir.exists(): continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.extensions and '_fullmask' not in p.stem:
                    fullmask = cls_dir / f'{p.stem}_fullmask.png'
                    if fullmask.exists(): self.samples.append((p, fullmask, label))
        if not self.samples:
            raise RuntimeError(f'No processed samples found in {self.root_dir}. Expected image + *_fullmask.png.')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = _safe_imread_color(str(img_path))
        mask = _safe_imread_gray(str(mask_path))
        image = cv2.resize(image, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)
        if self.augment and self.rng.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1]); mask = np.ascontiguousarray(mask[:, ::-1])
        if self.scales:
            patches, validity, coords, scale_ids = extract_lung_patches_multiscale(
                image, mask, self.scales, self.multiscale_output_size,
                self.max_patches_per_scale, self.augment, self.rng)
        else:
            patches, validity, coords, scale_ids, _ = extract_lung_patches(
                image, mask, self.patch_size, self.stride,
                self.coverage_threshold, self.max_patches, self.augment, self.rng)
        return (
            torch.from_numpy(patches).permute(0,3,1,2).contiguous(),
            torch.from_numpy(validity),
            torch.from_numpy(coords),
            torch.from_numpy(scale_ids),
            label,
        )
