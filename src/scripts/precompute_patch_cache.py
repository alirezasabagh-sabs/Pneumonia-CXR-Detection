from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from src.utils.config_loader import load_config
from src.patch_mil_data_loader import PatchMILDataset
from src.models.patch_mil import PatchMILClassifier

RELEVANCE_OVERLAP_THRESHOLD=0.3

def load_bbox_labels(csv_path):
    if not csv_path: return {}
    p=Path(csv_path)
    if not p.exists(): print(f'Warning: bbox CSV not found: {p}; continuing without bbox supervision.'); return {}
    out={}
    with p.open('r',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.setdefault(row['image_stem'],[]).append((float(row['x_min']),float(row['y_min']),float(row['x_max']),float(row['y_max'])))
    return out

def compute_patch_relevance(coords,canvas_size,patch_size,boxes):
    if not boxes: return np.full((len(coords),),-1.,dtype=np.float32)
    rel=np.zeros((len(coords),),dtype=np.float32); area=float(patch_size*patch_size)
    for i,(y_norm,x_norm) in enumerate(coords):
        px0=x_norm*canvas_size; py0=y_norm*canvas_size; px1=px0+patch_size; py1=py0+patch_size; best=0.
        for bx0,by0,bx1,by1 in boxes:
            ix0,iy0=max(px0,bx0),max(py0,by0); ix1,iy1=min(px1,bx1),min(py1,by1)
            if ix1>ix0 and iy1>iy0: best=max(best,((ix1-ix0)*(iy1-iy0))/area)
        rel[i]=1.0 if best>=RELEVANCE_OVERLAP_THRESHOLD else 0.0
    return rel

def build_cache(cfg,model,device,split,bbox_map,out_root):
    pm=cfg.patch_mil; use_ms=getattr(pm,'use_multiscale',False)
    ds=PatchMILDataset(str(cfg.paths.processed_root),split,pm.canvas_size,pm.patch_size,pm.stride,pm.coverage_threshold,
                       pm.max_patches,list(cfg.data.classes),list(cfg.data.image_extensions),False,cfg.project.seed,
                       [dict(s) for s in pm.scales] if use_ms else None,getattr(pm,'multiscale_output_size',64),getattr(pm,'max_patches_per_scale',50))
    out_dir=Path(out_root)/split; out_dir.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for idx in tqdm(range(len(ds)),desc=f'Cache {split}'):
            img_path,mask_path,label=ds.samples[idx]; out=out_dir/f'{img_path.stem}.npz'
            if out.exists(): continue
            patches,valid,coords,scale_ids,_=ds[idx]
            flat=patches.unsqueeze(0).to(device).view(-1,*patches.shape[1:])
            emb=model.encoder(flat).cpu().numpy().astype(np.float32)
            boxes=bbox_map.get(img_path.stem)
            rel=np.full((len(coords),),-1.,dtype=np.float32) if use_ms else compute_patch_relevance(coords,pm.canvas_size,pm.patch_size,boxes)
            np.savez_compressed(out,embeddings=emb,coords=coords.numpy().astype(np.float32),scale_ids=scale_ids.numpy().astype(np.int64),label=np.array(label),relevance=rel)

def main():
    cfg=load_config(); device=torch.device(cfg.device if torch.cuda.is_available() else 'cpu'); pm=cfg.patch_mil
    cache_root=Path(cfg.paths.patch_cache_root); bbox=load_bbox_labels(pm.rsna_bbox_labels_csv)
    model=PatchMILClassifier(pretrained_encoder=pm.pretrained_encoder,freeze_encoder=True,context_layers=0,encoder_type='torchxrayvision',txrv_input_size=pm.txrv_input_size).to(device); model.eval()
    for split in cfg.data.splits: build_cache(cfg,model,device,split,bbox,cache_root)

if __name__=='__main__': main()
