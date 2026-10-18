"""
engine.py -- shared training / prediction machinery used by train.py,
evaluate.py, cross_dataset_eval.py, dataset_probe.py and inference.py.
"""
import copy
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
import utils
from augmentations import FundusAugmentor
from dataset import FundusDataset, build_class_mapping
from losses import build_loss_fn, compute_class_weights, dataset_class_sample_weights
from metrics import safe_auroc, summarize, youden_threshold
from models import MultiBackboneFundusClassifier, count_trainable_params
from preprocessing import FundusPreprocessor


# --------------------------------------------------------------------------
def make_loader(df: pd.DataFrame, train: bool, sampler_weights=None, batch_size: int = None) -> DataLoader:
    ds = FundusDataset(df, build_class_mapping(), FundusPreprocessor(),
                       FundusAugmentor() if train else None)
    nw = config.NUM_WORKERS
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    bs = batch_size or config.BATCH_SIZE
    if train:
        if sampler_weights is not None:
            sampler = WeightedRandomSampler(torch.as_tensor(sampler_weights), num_samples=len(ds), replacement=True)
            return DataLoader(ds, batch_size=bs, sampler=sampler, drop_last=True, **kw)
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, **kw)
    return DataLoader(ds, batch_size=bs, shuffle=False, **kw)


@torch.no_grad()
def predict_df(model, df: pd.DataFrame, device, batch_size: int = 32, tta: bool = None) -> pd.DataFrame:
    """Returns a copy of df with column `prob_pos` = P(Glaucoma_or_Suspect).
    tta (default config.USE_TTA) averages each image with its horizontal flip."""
    tta = config.USE_TTA if tta is None else tta
    model.eval()
    df = df.reset_index(drop=True)
    loader = make_loader(df, train=False, batch_size=batch_size)
    probs = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=config.USE_AMP and device.type == "cuda"):
            logits = model(x)
            logits_f = model(torch.flip(x, dims=[3])) if tta else None
        p = torch.softmax(logits.float(), 1)[:, 1]
        if tta:
            p = 0.5 * (p + torch.softmax(logits_f.float(), 1)[:, 1])
        probs.append(p.cpu().numpy())
    out = df.copy()
    out["prob_pos"] = np.concatenate(probs)
    out["y"] = out["unified_class"].map(build_class_mapping()).astype(int)
    return out


def build_model_from_checkpoint(path: str, device):
    ck = utils.load_checkpoint_file(path, device)
    model = MultiBackboneFundusClassifier(backbone_names=ck.get("backbone_names"), pretrained=False).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------
def _lr_factor(epoch: int) -> float:
    w, total = config.WARMUP_EPOCHS, config.NUM_EPOCHS
    if epoch < w:
        return (epoch + 1) / (w + 1)
    prog = (epoch - w) / max(1, total - w)
    return max(0.02, 0.5 * (1 + math.cos(math.pi * prog)))


def build_optimizer(model) -> AdamW:
    """Built ONCE. Frozen params have grad=None and are skipped by AdamW, so
    un-freezing later does not require rebuilding (keeps optimizer state and
    the LR schedule intact)."""
    bb = [p for m in model.backbones.values() for p in m.parameters()]
    ids = {id(p) for p in bb}
    head = [p for p in model.parameters() if id(p) not in ids]
    return AdamW([{"params": bb, "lr": config.BACKBONE_LR},
                  {"params": head, "lr": config.LEARNING_RATE}], weight_decay=config.WEIGHT_DECAY)


def run_epoch(model, loader, loss_fn, optimizer, device, scaler, train: bool, accum: int = 1, ema=None) -> Dict:
    model.train(train)
    total, n = 0.0, 0
    probs, targets = [], []
    steps = len(loader)
    if train:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=config.USE_AMP and device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            if train:
                scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0 or step + 1 == steps:      # also flush the tail
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
            total += loss.item() * x.size(0)
            n += x.size(0)
            probs.append(torch.softmax(logits.detach().float(), 1)[:, 1].cpu().numpy())
            targets.append(y.cpu().numpy())
    p, t = np.concatenate(probs), np.concatenate(targets)
    return {"loss": total / max(n, 1), "auroc": safe_auroc(t, p),
            "balanced_acc": float(balanced_accuracy_score(t, p >= 0.5)) if len(np.unique(t)) == 2 else float("nan")}


# --------------------------------------------------------------------------
def run_training(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str,
                 epochs: Optional[int] = None, tag: str = "main") -> Dict:
    device = utils.get_device()
    utils.set_seed(config.RANDOM_SEED)
    os.makedirs(out_dir, exist_ok=True)
    epochs = epochs or config.NUM_EPOCHS
    c2i = build_class_mapping()

    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    if len(train_df) < config.BATCH_SIZE * 2:
        raise ValueError(f"Only {len(train_df)} training images -- too few for batch size {config.BATCH_SIZE}.")
    train_labels = train_df["unified_class"].map(c2i).to_numpy()

    weights = dataset_class_sample_weights(train_df) if config.IMBALANCE_STRATEGY == "dataset_class_sampler" else None
    if config.IMBALANCE_STRATEGY == "weighted_sampler":
        cw = compute_class_weights(train_labels, config.NUM_CLASSES).numpy()
        weights = cw[train_labels]
    train_loader = make_loader(train_df, True, weights)
    val_loader = make_loader(val_df, False)

    model = MultiBackboneFundusClassifier().to(device)
    print(f"[{tag}] backbones={model.backbone_names} | trainable (frozen phase)={count_trainable_params(model):,} "
          f"| train={len(train_df)} val={len(val_df)}")
    loss_fn = build_loss_fn(config.IMBALANCE_STRATEGY, train_labels, config.NUM_CLASSES, config.FOCAL_LOSS_GAMMA,
                            config.LABEL_SMOOTHING).to(device)
    optimizer = build_optimizer(model)
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device.type == "cuda")
    stopper = utils.EarlyStopping(config.EARLY_STOPPING_PATIENCE)
    ema = utils.ModelEma(model, config.EMA_DECAY) if config.USE_EMA else None

    history, best_state, best_epoch, best_metric = [], None, 0, -1.0
    extra = {"backbone_names": model.backbone_names, "class_names": config.CLASS_NAMES,
             "image_size": config.IMAGE_SIZE}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if epoch == config.FREEZE_EPOCHS + 1:
            model.set_backbones_trainable(True)
            model.unfreeze_last_n_tensors(config.UNFREEZE_LAST_N_TENSORS)
            print(f"[{tag}] epoch {epoch}: backbones un-frozen ({count_trainable_params(model):,} trainable)")
        tr = run_epoch(model, train_loader, loss_fn, optimizer, device, scaler, True, config.GRAD_ACCUM_STEPS, ema)
        # validate + checkpoint the EMA weights once warm (raw weights before that)
        eval_model = ema.module if (ema is not None and ema.step >= config.EMA_WARMUP_STEPS) else model
        va = run_epoch(eval_model, val_loader, loss_fn, optimizer, device, scaler, False)
        scheduler.step()

        metric = va["auroc"] if not np.isnan(va["auroc"]) else va["balanced_acc"]
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_auroc": tr["auroc"],
                        "val_loss": va["loss"], "val_auroc": va["auroc"], "val_balanced_acc": va["balanced_acc"],
                        "lr_head": optimizer.param_groups[1]["lr"]})
        print(f"[{tag}] ep {epoch:02d}/{epochs} {time.time()-t0:5.0f}s | train loss {tr['loss']:.4f} "
              f"auc {tr['auroc']:.3f} | val loss {va['loss']:.4f} auc {va['auroc']:.3f} bal-acc {va['balanced_acc']:.3f}")
        if stopper.step(metric):
            best_state, best_epoch, best_metric = copy.deepcopy(eval_model.state_dict()), epoch, metric
            utils.save_checkpoint(eval_model, os.path.join(out_dir, "best_model.pth"), epoch, metric, extra)
            print(f"[{tag}]   -> new best val {config.EARLY_STOPPING_METRIC}={metric:.4f}")
        utils.save_checkpoint(eval_model, os.path.join(out_dir, "last_model.pth"), epoch, metric, extra)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
        if stopper.should_stop:
            print(f"[{tag}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid epoch.")
    model.load_state_dict(best_state)

    # Operating threshold is chosen on VALIDATION only (Youden J), never on test.
    vp = predict_df(model, val_df, device)
    thr = youden_threshold(vp["y"], vp["prob_pos"]) if vp["y"].nunique() == 2 else 0.5
    utils.save_json({"threshold": thr, "method": "youden_on_val", "best_epoch": best_epoch,
                     "val": summarize(vp["y"], vp["prob_pos"], thr)},
                    os.path.join(out_dir, "operating_threshold.json"))
    print(f"[{tag}] best epoch {best_epoch} | val {config.EARLY_STOPPING_METRIC}={best_metric:.4f} | threshold={thr:.3f}")
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_metric": best_metric, "threshold": thr, "device": device}
