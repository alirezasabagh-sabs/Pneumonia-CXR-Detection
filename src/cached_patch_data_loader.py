from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

class CachedPatchDataset(Dataset):
    def __init__(self, cache_root, split, max_patches, augment=False, seed=42):
        self.cache_dir=Path(cache_root)/split; self.max_patches=int(max_patches); self.augment=augment; self.seed=int(seed)
        self.files=sorted(self.cache_dir.glob('*.npz'))
        if not self.files: raise RuntimeError(f'No cache found in {self.cache_dir}. Run precompute_patch_cache.py first.')
    def __len__(self): return len(self.files)
    def _rng(self, idx):
        w=get_worker_info(); wid=0 if w is None else w.id
        return np.random.default_rng((self.seed+1000003*wid+9176*idx)&0xffffffff)
    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            emb=data['embeddings']; coords=data['coords']; scale=data['scale_ids']; label=int(data['label'])
            rel=data['relevance'] if 'relevance' in data.files else np.full((len(emb),),-1.,dtype=np.float32)
        n=len(emb)
        if n==0: raise RuntimeError(f'Empty cache: {self.files[idx]}')
        rng=self._rng(idx)
        if n>=self.max_patches:
            sel=rng.choice(n,self.max_patches,replace=False) if self.augment else np.arange(self.max_patches)
        else:
            extra=rng.integers(0,n,size=self.max_patches-n); sel=np.concatenate([np.arange(n),extra])
        valid=np.ones(self.max_patches,dtype=np.float32)
        return torch.from_numpy(emb[sel]).float(),torch.from_numpy(valid),torch.from_numpy(coords[sel]).float(),torch.from_numpy(scale[sel]).long(),torch.from_numpy(rel[sel]).float(),label
