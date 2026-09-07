from __future__ import annotations
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = max(1, int(total_epochs * steps_per_epoch))
    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        p = (step-warmup_steps) / max(1, total_steps-warmup_steps)
        p = min(max(p,0.0),1.0)
        return 0.5*(1+math.cos(math.pi*p))
    return LambdaLR(optimizer, lr_lambda)

def calibrate_threshold_for_sensitivity(probs, labels, target_sensitivity):
    probs=np.asarray(probs, dtype=np.float64); labels=np.asarray(labels).astype(int)
    if not 0.0 <= float(target_sensitivity) <= 1.0:
        raise ValueError("target_sensitivity must be between 0 and 1.")
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape.")
    positives=labels==1
    if positives.sum()==0: raise ValueError('Validation set contains no positive samples.')
    candidates=np.unique(np.concatenate(([0.0], probs, [1.0])))
    best_t=0.0; best_spec=-1.0
    for t in candidates:
        pred=(probs>=t).astype(int)
        tp=np.sum((pred==1)&positives); fn=np.sum((pred==0)&positives)
        sens=tp/max(1,tp+fn)
        if sens < target_sensitivity: continue
        neg=~positives; tn=np.sum((pred==0)&neg); fp=np.sum((pred==1)&neg)
        spec=tn/max(1,tn+fp)
        if spec>best_spec: best_spec=float(spec); best_t=float(t)
    return best_t
