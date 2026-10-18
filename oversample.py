"""
oversample.py -- offline minority-class oversampling (TRAIN split only).

    python build_merged_dataset.py --cache --oversample

Writes new augmented image files (one random FundusAugmentor draw each) for
minority-class TRAIN images and appends them to the index with
is_synthetic=True / source_image_path=<original>. Copies inherit the source
patient_id and only train rows are ever copied, so val/test and the
patient-level split are untouched. Per `domain`, each class is topped up
toward OVERSAMPLE_TARGET_RATIO x the domain's majority count, at most
OVERSAMPLE_MAX_COPIES_PER_IMAGE per source image. Re-runs reuse existing files.
"""
import hashlib
import os
import random

import cv2
import numpy as np
import pandas as pd

import config
from augmentations import FundusAugmentor
from preprocessing import FundusPreprocessor


def _out_path(cache_dir, src, k):
    h = hashlib.sha1(f"{os.path.abspath(src).lower()}::{k}".encode()).hexdigest()[:20]
    return os.path.join(cache_dir, "augmented", h + ".png")


def _make_copy(pre, aug, src, k, seed):
    out = _out_path(pre.cache_dir, src, k)
    if os.path.exists(out):
        return out
    img = pre.load_processed(src)
    if img is None:
        return None
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp.png"
    cv2.imwrite(tmp, cv2.cvtColor(aug(img), cv2.COLOR_RGB2BGR))
    os.replace(tmp, out)
    return out


def oversample_train_split(df, target_ratio=None, max_copies=None):
    target_ratio = config.OVERSAMPLE_TARGET_RATIO if target_ratio is None else target_ratio
    max_copies = config.OVERSAMPLE_MAX_COPIES_PER_IMAGE if max_copies is None else max_copies
    df = df.copy()
    if "is_synthetic" not in df.columns:
        df["is_synthetic"], df["source_image_path"] = False, None
    pre, aug = FundusPreprocessor(), FundusAugmentor()
    tasks = []
    for domain, d in df[df["split"] == "train"].groupby("domain"):
        counts = d["unified_class"].value_counts()
        if len(counts) < 2:
            continue
        target = int(round(counts.max() * target_ratio))
        for cls, n in counts.items():
            need = target - int(n)
            if need <= 0:
                continue
            rows = d[d["unified_class"] == cls]
            per = min(max_copies, -(-need // len(rows)))
            made = 0
            for _, r in rows.iterrows():
                for k in range(per):
                    if made >= need:
                        break
                    seed = int(hashlib.sha1(f"{r['image_path']}::{k}".encode()).hexdigest()[:8], 16)
                    tasks.append((r.to_dict(), k, seed)); made += 1
            print(f"[oversample] {domain:16s} {cls:22s} {int(n):4d} -> {target:4d} (+{made}, <= {per}/image)")
    new = []
    for r, k, seed in tasks:
        p = _make_copy(pre, aug, r["image_path"], k, seed)
        if p is None:
            print(f"[WARN] oversample: unreadable source {r['image_path']}"); continue
        new.append({**r, "image_path": p, "is_synthetic": True, "source_image_path": r["image_path"]})
    out = pd.concat([df, pd.DataFrame(new)], ignore_index=True) if new else df
    print(f"[oversample] train {int((df.split == 'train').sum())} -> {int((out.split == 'train').sum())} "
          f"(+{len(new)} synthetic); val/test untouched")
    return out
