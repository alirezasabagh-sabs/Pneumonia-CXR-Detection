"""Cached Patch-MIL training entry point."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.config_loader import load_config
from src.cached_patch_data_loader import CachedPatchDataset
from src.models.patch_mil import PatchMILClassifier
from src.losses.asymmetric_focal_loss import build_loss
from src.train import make_warmup_cosine_scheduler, calibrate_threshold_for_sensitivity

def loaders(cfg,cache_root):
    return {s:DataLoader(CachedPatchDataset(cache_root,s,cfg.patch_mil.max_patches,s=='train',cfg.project.seed),batch_size=cfg.data.batch_size,shuffle=(s=='train'),num_workers=cfg.data.num_workers,pin_memory=torch.cuda.is_available(),drop_last=(s=='train')) for s in cfg.data.splits}

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); p=[]; y=[]
    for e,v,_,sid,mp,_,lab in tqdm(loader,desc='eval',leave=False):
        e=e.to(device); v=v.to(device); sid=sid.to(device); mp=mp.to(device)
        logits=model.forward_from_tokens(e,v,sid,mask_prior=mp); p.extend(torch.sigmoid(logits).cpu().numpy().tolist()); y.extend(lab.numpy().tolist())
    p=np.asarray(p); y=np.asarray(y); return {'auc':float(roc_auc_score(y,p)),'probs':p,'labels':y}

def train():
    cfg=load_config(); pm=cfg.patch_mil; torch.manual_seed(cfg.project.seed); np.random.seed(cfg.project.seed)
    device=torch.device(cfg.device if torch.cuda.is_available() else 'cpu'); cache=Path(cfg.paths.patch_cache_root); ckpt=Path(cfg.paths.patch_mil_checkpoint_dir)/'cached'; ckpt.mkdir(parents=True,exist_ok=True)
    ls=loaders(cfg,cache)
    model=PatchMILClassifier(hidden_size=1024,attn_hidden=pm.attention_pooling_hidden,head_dims=list(pm.head_dims),dropout=pm.dropout,context_layers=getattr(pm,'context_layers',0),context_heads=getattr(pm,'context_heads',8),encoder_type='torchxrayvision',mask_strength=float(getattr(pm,'mask_strength',0.0)),hard_filter_threshold=getattr(pm,'hard_filter_threshold',None),skip_encoder_build=True).to(device)
    loss_fn=build_loss(cfg); opt=AdamW(model.get_param_groups(cfg.training.lr_backbone,cfg.training.lr_head),weight_decay=cfg.training.weight_decay); sched=make_warmup_cosine_scheduler(opt,cfg.training.warmup_epochs,cfg.training.epochs,len(ls['train']))
    best=-np.inf; bad=0; best_path=ckpt/'best_model.pt'; eval_every=max(1,int(getattr(pm,'eval_every_n_epochs',1)))
    bbox_weight=float(getattr(pm,'bbox_lambda',0.0)); bbox_pos=float(getattr(pm,'bbox_pos_weight',5.0)); bbox_loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bbox_pos,device=device))
    for epoch in range(cfg.training.epochs):
        model.train();
        for e,v,c,sid,mp,rel,y in tqdm(ls['train'],desc=f'Epoch {epoch+1}',leave=False):
            e=e.to(device); v=v.to(device); c=c.to(device); sid=sid.to(device); mp=mp.to(device); rel=rel.to(device); y=y.to(device).float(); opt.zero_grad(set_to_none=True)
            logits,attn,critical,inst=model.forward_with_attention_and_instance_from_tokens(e,v,sid,mask_prior=mp); main=loss_fn(logits,y); cont=model.compute_contiguity_loss(attn,c); safe=attn.clamp_min(1e-12); ent=-(safe*torch.log(safe)).sum(1).mean(); aux=loss_fn(critical,y)
            m=rel>=0; bxl=bbox_loss_fn(inst[m],rel[m]) if m.any() else inst.new_zeros(()); warm=int(getattr(pm,'entropy_lambda_warmup_epochs',0)); ramp=min(1.,(epoch+1)/warm) if warm else 1.; total=main+float(getattr(pm,'contiguity_lambda',0))*cont+float(getattr(pm,'entropy_lambda',0))*ramp*ent+float(getattr(pm,'instance_aux_lambda',0))*aux+bbox_weight*bxl
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.grad_clip_norm); opt.step(); sched.step()
        if ((epoch+1)%eval_every==0) or epoch==cfg.training.epochs-1:
            val=evaluate(model,ls['val'],device); print(f'Epoch {epoch+1}: val_auc={val["auc"]:.4f}')
            if val['auc']>best: best=val['auc']; bad=0; torch.save(model.state_dict(),best_path)
            else:
                bad+=1
                if bad>=cfg.training.early_stopping_patience: break
    model.load_state_dict(torch.load(best_path,map_location=device)); val=evaluate(model,ls['val'],device); threshold=calibrate_threshold_for_sensitivity(val['probs'],val['labels'],cfg.training.target_sensitivity)
    test=evaluate(model,ls['test'],device); pred=(test['probs']>=threshold).astype(int); y=test['labels']; tp=((pred==1)&(y==1)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum(); fp=((pred==1)&(y==0)).sum(); result={'auc':float(test['auc']),'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),'threshold':float(threshold)}
    (ckpt/'calibration.json').write_text(json.dumps({'target_sensitivity':float(cfg.training.target_sensitivity),'calibrated_threshold':float(threshold),'val_auc':float(val['auc'])},indent=2),encoding='utf-8')
    (ckpt/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': train()
