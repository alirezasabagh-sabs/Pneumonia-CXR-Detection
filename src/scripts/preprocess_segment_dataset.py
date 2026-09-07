"""Precompute lung masks and full-mask PNGs for Patch-MIL."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from src.utils.config_loader import load_config
from src.models.lung_segmenter import LungSegmenter

def safe_imread(path):
    data=np.fromfile(str(path),dtype=np.uint8); img=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f'Could not read: {path}')
    return img

def main(config_path=None):
    cfg=load_config(config_path); data_root=Path(cfg.paths.data_root); processed=Path(cfg.paths.processed_root); seg=LungSegmenter()
    for split in cfg.data.splits:
        for cls in cfg.data.classes:
            src=data_root/split/cls; dst=processed/split/cls; dst.mkdir(parents=True,exist_ok=True)
            if not src.exists(): continue
            paths=[p for p in src.rglob('*') if p.suffix.lower() in set(cfg.data.image_extensions)]
            for p in tqdm(paths,desc=f'{split}/{cls}'):
                out=dst/p.name; mask_out=dst/f'{p.stem}_fullmask.png'
                if out.exists() and mask_out.exists(): continue
                mask=seg.predict_mask(str(p),clean=cfg.segmentation.clean_mask,remove_black_borders=True,mask_binary_threshold=cfg.segmentation.mask_binary_threshold,use_convex_hull=cfg.segmentation.use_convex_hull)
                img=seg.crop_black_borders(safe_imread(p)); img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); cv2.imwrite(str(out),cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR))
                full=(cv2.resize(mask.astype(np.float32),(img_rgb.shape[1],img_rgb.shape[0]),interpolation=cv2.INTER_LINEAR)>0.5).astype(np.uint8)*255; cv2.imwrite(str(mask_out),full)
if __name__=='__main__': main()
