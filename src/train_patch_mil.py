from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.utils.config_loader import load_config
from src.patch_mil_data_loader import PatchMILDataset
from src.models.patch_mil import PatchMILClassifier
from src.losses.asymmetric_focal_loss import build_loss
from src.train import make_warmup_cosine_scheduler, calibrate_threshold_for_sensitivity

def build_dataloaders(cfg):
    pm=cfg.patch_mil
    datasets={}
    for split in cfg.data.splits:
        datasets[split]=PatchMILDataset(
            root_dir=str(cfg.paths.processed_root), split=split,
            canvas_size=pm.canvas_size, patch_size=pm.patch_size,
            stride=pm.stride, coverage_threshold=pm.coverage_threshold,
            max_patches=pm.max_patches, classes=list(cfg.data.classes),
            extensions=list(cfg.data.image_extensions), augment=(split=='train'), seed=cfg.project.seed,
            scales=[dict(s) for s in pm.scales] if getattr(pm,'use_multiscale',False) else None,
            multiscale_output_size=getattr(pm,'multiscale_output_size',64),
            max_patches_per_scale=getattr(pm,'max_patches_per_scale',50),
        )
    return {split:DataLoader(ds,batch_size=cfg.data.batch_size,shuffle=(split=='train'),
                            num_workers=cfg.data.num_workers,pin_memory=torch.cuda.is_available(),
                            drop_last=(split=='train')) for split,ds in datasets.items()}

@torch.no_grad()
def evaluate(model, loader, device, desc='eval'):
    model.eval(); probs=[]; labels=[]; amp=cfg_amp=True
    for patches,validity,_,scale_ids,y in tqdm(loader,desc=desc,leave=False):
        patches=patches.to(device); validity=validity.to(device); scale_ids=scale_ids.to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type=='cuda')):
            logits=model(patches,validity,scale_ids)
        probs.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist()); labels.extend(y.numpy().tolist())
    probs=np.asarray(probs); labels=np.asarray(labels)
    auc=roc_auc_score(labels,probs)
    pred=(probs>=0.5).astype(int); tp=((pred==1)&(labels==1)).sum(); fn=((pred==0)&(labels==1)).sum()
    return {'auc':float(auc),'sensitivity_at_0.5':float(tp/max(1,tp+fn)),'probs':probs,'labels':labels}

def train():
    cfg=load_config(); pm=cfg.patch_mil
    torch.manual_seed(cfg.project.seed); np.random.seed(cfg.project.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.project.seed)
    device=torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    ckpt=Path(cfg.paths.patch_mil_checkpoint_dir)/'raw'; ckpt.mkdir(parents=True,exist_ok=True)
    loaders=build_dataloaders(cfg)
    freeze_epochs=int(pm.freeze_encoder_epochs); partial=int(getattr(pm,'partial_unfreeze_last_n_blocks',0))
    model=PatchMILClassifier(attn_hidden=pm.attention_pooling_hidden,head_dims=list(pm.head_dims),dropout=pm.dropout,
                             pretrained_encoder=pm.pretrained_encoder,freeze_encoder=(freeze_epochs>0),
                             context_layers=getattr(pm,'context_layers',0),context_heads=getattr(pm,'context_heads',8),
                             encoder_type=pm.encoder_type,num_scales=1,txrv_input_size=getattr(pm,'txrv_input_size',128)).to(device)
    loss_fn=build_loss(cfg)
    opt=AdamW(model.get_param_groups(cfg.training.lr_backbone,cfg.training.lr_head),weight_decay=cfg.training.weight_decay)
    sched=make_warmup_cosine_scheduler(opt,cfg.training.warmup_epochs,cfg.training.epochs,len(loaders['train']))
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.training.mixed_precision and device.type=='cuda')
    best=-np.inf; bad=0; eval_every=max(1,int(getattr(pm,'eval_every_n_epochs',1))); best_path=ckpt/'best_model.pt'
    for epoch in range(cfg.training.epochs):
        if epoch==freeze_epochs:
            model.set_backbone_partial_trainable(partial) if partial>0 else model.set_backbone_trainable(True)
        model.train(); t=time.time(); totals=[0.0,0.0,0.0,0.0]
        for patches,validity,coords,scale_ids,labels in tqdm(loaders['train'],desc=f'Epoch {epoch+1} train',leave=False):
            patches=patches.to(device); validity=validity.to(device); coords=coords.to(device); scale_ids=scale_ids.to(device); labels=labels.to(device).float()
            opt.zero_grad(set_to_none=True)
            amp=cfg.training.mixed_precision and device.type=='cuda'
            with torch.autocast(device_type=device.type,enabled=amp):
                logits,attn=model.forward_with_attention(patches,validity,scale_ids)
                ce=loss_fn(logits,labels); cont=model.compute_contiguity_loss(attn,coords)
                safe=attn.clamp_min(1e-12); ent=-(safe*torch.log(safe)).sum(dim=1).mean()
                warm=int(getattr(pm,'entropy_lambda_warmup_epochs',0)); ramp=min(1.0,(epoch+1)/warm) if warm>0 else 1.0
                loss=ce+float(getattr(pm,'contiguity_lambda',0.0))*cont+float(getattr(pm,'entropy_lambda',0.0))*ramp*ent
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.grad_clip_norm); scaler.step(opt); scaler.update(); sched.step()
            totals[0]+=loss.item(); totals[1]+=ce.item(); totals[2]+=cont.item(); totals[3]+=ent.item()
        if ((epoch+1)%eval_every==0) or epoch==cfg.training.epochs-1:
            val=evaluate(model,loaders['val'],device,desc=f'Epoch {epoch+1} val')
            print(f'Epoch {epoch+1}: loss={totals[0]/len(loaders["train"]):.4f} val_auc={val["auc"]:.4f} val_sens@0.5={val["sensitivity_at_0.5"]:.4f} time={time.time()-t:.1f}s')
            if val['auc']>best:
                best=val['auc']; bad=0; torch.save(model.state_dict(),best_path)
            else:
                bad+=1
                if bad>=cfg.training.early_stopping_patience: break
    if not best_path.exists(): raise RuntimeError('No best Patch-MIL checkpoint was produced.')
    model.load_state_dict(torch.load(best_path,map_location=device)); val=evaluate(model,loaders['val'],device,'Final val')
    threshold=calibrate_threshold_for_sensitivity(val['probs'],val['labels'],cfg.training.target_sensitivity)
    (ckpt/'calibration.json').write_text(json.dumps({'target_sensitivity':float(cfg.training.target_sensitivity),'calibrated_threshold':float(threshold),'val_auc':float(val['auc'])},indent=2),encoding='utf-8')
    test=evaluate(model,loaders['test'],device,'Final test'); pred=(test['probs']>=threshold).astype(int); y=test['labels']
    tp=((pred==1)&(y==1)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum(); fp=((pred==1)&(y==0)).sum()
    sens=tp/max(1,tp+fn); spec=tn/max(1,tn+fp)
    result={'auc':float(test['auc']),'sensitivity':float(sens),'specificity':float(spec),'threshold':float(threshold)}
    (ckpt/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': train()
