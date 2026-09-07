"""Train the raw-image Patch-MIL pneumonia classifier."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.asymmetric_focal_loss import build_loss
from src.models.patch_mil import PatchMILClassifier
from src.patch_mil_data_loader import PatchMILDataset
from src.train import calibrate_threshold_for_sensitivity, make_warmup_cosine_scheduler
from src.utils.config_loader import load_config


def build_dataloaders(cfg):
    pm = cfg.patch_mil
    datasets = {}
    for split in cfg.data.splits:
        datasets[split] = PatchMILDataset(
            root_dir=str(cfg.paths.processed_root), split=split,
            canvas_size=pm.canvas_size, patch_size=pm.patch_size, stride=pm.stride,
            coverage_threshold=pm.coverage_threshold, max_patches=pm.max_patches,
            classes=list(cfg.data.classes), extensions=list(cfg.data.image_extensions),
            augment=(split == "train"), seed=cfg.project.seed,
            scales=[dict(s) for s in pm.scales] if getattr(pm, "use_multiscale", False) else None,
            multiscale_output_size=getattr(pm, "multiscale_output_size", 64),
            max_patches_per_scale=getattr(pm, "max_patches_per_scale", 50),
            mask_prior_type=getattr(pm, "mask_prior_type", "distance"),
        )
    return {
        split: DataLoader(
            ds, batch_size=cfg.data.batch_size, shuffle=(split == "train"),
            num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    model.eval()
    probs, labels = [], []
    amp = device.type == "cuda"
    for patches, validity, _coords, scale_ids, mask_prior, y in tqdm(loader, desc=desc, leave=False):
        patches = patches.to(device)
        validity = validity.to(device)
        scale_ids = scale_ids.to(device)
        mask_prior = mask_prior.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(patches, validity, scale_ids, mask_prior=mask_prior)
        probs.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    auc = roc_auc_score(labels, probs)
    pred = (probs >= 0.5).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    return {"auc": float(auc), "sensitivity_at_0.5": float(tp / max(1, tp + fn)), "probs": probs, "labels": labels}


def _save_architecture(model, pm, path):
    info = {
        "encoder_type": pm.encoder_type, "encoder_feature_dim": model.hidden_size,
        "attention_hidden": pm.attention_pooling_hidden, "head_dims": list(pm.head_dims),
        "dropout": pm.dropout, "context_layers": getattr(pm, "context_layers", 0),
        "context_heads": getattr(pm, "context_heads", 8), "txrv_input_size": pm.txrv_input_size,
        "canvas_size": pm.canvas_size, "patch_size": pm.patch_size, "stride": pm.stride,
        "coverage_threshold": pm.coverage_threshold, "max_patches": pm.max_patches,
        "partial_unfreeze_last_n_blocks": getattr(pm, "partial_unfreeze_last_n_blocks", 0),
        "mask_prior_type": getattr(pm, "mask_prior_type", "distance"),
    }
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def train():
    cfg = load_config()
    pm = cfg.patch_mil
    torch.manual_seed(cfg.project.seed)
    np.random.seed(cfg.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.project.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Patch-MIL | encoder={pm.encoder_type}", flush=True)

    ckpt_dir = Path(cfg.paths.patch_mil_checkpoint_dir) / "raw"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_dataloaders(cfg)
    freeze_epochs = int(pm.freeze_encoder_epochs)
    partial_unfreeze = int(getattr(pm, "partial_unfreeze_last_n_blocks", 0))

    model = PatchMILClassifier(
        attn_hidden=pm.attention_pooling_hidden, head_dims=list(pm.head_dims), dropout=pm.dropout,
        pretrained_encoder=pm.pretrained_encoder, freeze_encoder=(freeze_epochs > 0),
        context_layers=getattr(pm, "context_layers", 0), context_heads=getattr(pm, "context_heads", 8),
        encoder_type=pm.encoder_type, txrv_input_size=pm.txrv_input_size,
        mask_strength=float(getattr(pm, "mask_strength", 0.0)),
        hard_filter_threshold=getattr(pm, "hard_filter_threshold", None),
    ).to(device)

    loss_fn = build_loss(cfg)
    optimizer = AdamW(
        model.get_param_groups(cfg.training.lr_backbone, cfg.training.lr_head),
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer, cfg.training.warmup_epochs, cfg.training.epochs, len(loaders["train"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision and device.type == "cuda")
    best_auc, bad_epochs = -np.inf, 0
    eval_every = max(1, int(getattr(pm, "eval_every_n_epochs", 1)))
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(cfg.training.epochs):
        if epoch == freeze_epochs:
            if partial_unfreeze > 0:
                model.set_backbone_partial_trainable(partial_unfreeze)
            else:
                model.set_backbone_trainable(True)

        model.train()
        start = time.time()
        sums = {"loss": 0.0, "classification": 0.0, "contiguity": 0.0, "entropy": 0.0}
        for patches, validity, coords, scale_ids, mask_prior, labels in tqdm(
            loaders["train"], desc=f"Epoch {epoch + 1} train", leave=False
        ):
            patches, validity = patches.to(device), validity.to(device)
            coords, scale_ids, mask_prior = coords.to(device), scale_ids.to(device), mask_prior.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            amp = cfg.training.mixed_precision and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, attn = model.forward_with_attention(patches, validity, scale_ids, mask_prior=mask_prior)
                classification = loss_fn(logits, labels)
                contiguity = model.compute_contiguity_loss(attn, coords)
                safe = attn.clamp_min(1e-12)
                entropy = -(safe * torch.log(safe)).sum(dim=1).mean()
                warm = int(getattr(pm, "entropy_lambda_warmup_epochs", 0))
                ramp = min(1.0, (epoch + 1) / warm) if warm else 1.0
                total = (classification + float(getattr(pm, "contiguity_lambda", 0.0)) * contiguity +
                         float(getattr(pm, "entropy_lambda", 0.0)) * ramp * entropy)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            sums["loss"] += total.item(); sums["classification"] += classification.item()
            sums["contiguity"] += contiguity.item(); sums["entropy"] += entropy.item()

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch == cfg.training.epochs - 1)
        if not should_eval:
            print(f"Epoch {epoch + 1}: train={sums['loss'] / len(loaders['train']):.4f} | validation skipped", flush=True)
            continue
        val = evaluate(model, loaders["val"], device, desc=f"Epoch {epoch + 1} val")
        mean_loss = sums["loss"] / len(loaders["train"])
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | loss={mean_loss:.4f} | val_auc={val['auc']:.4f} | time={time.time()-start:.1f}s", flush=True)
        if val["auc"] > best_auc:
            best_auc, bad_epochs = val["auc"], 0
            torch.save(model.state_dict(), best_path)
            _save_architecture(model, pm, ckpt_dir / "model_arch.json")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.training.early_stopping_patience:
                break

    if not best_path.exists():
        raise RuntimeError("No best Patch-MIL checkpoint was produced.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val = evaluate(model, loaders["val"], device, desc="Final val")
    threshold = calibrate_threshold_for_sensitivity(val["probs"], val["labels"], cfg.training.target_sensitivity)
    (ckpt_dir / "calibration.json").write_text(
        json.dumps({"target_sensitivity": float(cfg.training.target_sensitivity), "calibrated_threshold": float(threshold), "val_auc": val["auc"]}, indent=2),
        encoding="utf-8",
    )
    test = evaluate(model, loaders["test"], device, desc="Final test")
    pred = (test["probs"] >= threshold).astype(int); y = test["labels"]
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    result = {"auc": test["auc"], "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)), "threshold": float(threshold)}
    (ckpt_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    train()
