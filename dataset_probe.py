"""
dataset_probe.py -- shortcut / confound diagnostics.

    python dataset_probe.py                 # metadata probe (no torch, no checkpoint)
    python dataset_probe.py --embeddings    # + probe on the trained model's embeddings

Metadata probe: can trivial image properties (original resolution, file size,
FOV colour/brightness/sharpness statistics) predict the LABEL?
  * pooled AUROC well above chance but within-dataset AUROC ~0.5  => the label is
    entangled with the dataset (shortcut risk).  Expected here because Chaksu
    and PAPILA have different positive rates.
  * within-dataset AUROC clearly above 0.5 => cameras/sites leak label info even
    inside one dataset (investigate).
Also reports how easily the DATASET itself is recognised (expected ~1.0).

Embedding probe: can a linear classifier recover dataset / camera from the
trained network's penultimate features?  High = the network encodes camera
identity; compare with label AUROC and watch the cross-dataset results.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from preprocessing import FundusPreprocessor


def meta_features(df: pd.DataFrame) -> pd.DataFrame:
    pre = FundusPreprocessor()
    rows = []
    for p in df["image_path"]:
        with Image.open(p) as im:
            w, h = im.size
        img = pre.load_processed(p)
        fov = img.max(axis=2) > 0
        f = img.astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rows.append({"orig_w": w, "orig_h": h, "aspect": w / h, "file_kb": os.path.getsize(p) / 1024,
                     "fov_frac": fov.mean(),
                     **{f"mean_{c}": f[..., i][fov].mean() for i, c in enumerate("rgb")},
                     **{f"std_{c}": f[..., i][fov].std() for i, c in enumerate("rgb")},
                     "sharp": cv2.Laplacian(gray, cv2.CV_64F).var()})
    return pd.DataFrame(rows, index=df.index)


def _cv_auc(X, y, groups, model, multiclass=False):
    n_splits = max(2, min(5, pd.Series(groups).nunique()))
    cv = GroupKFold(n_splits=n_splits)
    if multiclass:
        pr = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")
        classes = np.unique(y)
        return float(np.mean(np.argmax(pr, 1) == np.searchsorted(classes, y)))
    pr = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pr)) if len(np.unique(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(config.MERGED_INDEX_PATH)
    df = df[df["dataset_name"].isin(config.ACTIVE_DATASETS)].reset_index(drop=True)
    if "is_synthetic" in df.columns:
        df = df[~df["is_synthetic"].astype(bool)].reset_index(drop=True)
    y = (df["unified_class"] == config.POSITIVE_CLASS).astype(int).to_numpy()
    g = df["patient_id"].to_numpy()

    print("Computing metadata features (uses the preprocessing cache)...")
    X = meta_features(df)
    gb = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3)

    print("\n=== METADATA PROBE: predict LABEL from trivial image properties ===")
    print(f"pooled                 AUROC {_cv_auc(X.values, y, g, gb):.3f}   (chance = 0.5)")
    for ds, idx in df.groupby("dataset_name").groups.items():
        idx = np.array(list(idx))
        print(f"within {ds:16s} AUROC {_cv_auc(X.values[idx], y[idx], g[idx], gb):.3f}")

    print("\n=== Can the DATASET be recognised from the same properties? (accuracy) ===")
    print(f"dataset  : {_cv_auc(X.values, df['dataset_name'].to_numpy(), g, gb, multiclass=True):.3f}")
    print(f"camera   : {_cv_auc(X.values, df['domain'].to_numpy(), g, gb, multiclass=True):.3f}")
    X.assign(label=y, dataset=df["dataset_name"], domain=df["domain"]).to_csv(
        os.path.join(config.OUTPUT_DIR, "metadata_probe_features.csv"), index=False)

    if args.embeddings:
        import torch
        import utils
        from engine import build_model_from_checkpoint, make_loader
        device = utils.get_device()
        model, _ = build_model_from_checkpoint(config.BEST_MODEL_PATH, device)
        loader = make_loader(df, train=False, batch_size=32)
        embs = []
        with torch.no_grad():
            for x, _, _ in loader:
                embs.append(model.embed(x.to(device)).float().cpu().numpy())
        E = np.concatenate(embs)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        print("\n=== EMBEDDING PROBE (trained model, all splits) ===")
        print(f"label   AUROC : {_cv_auc(E, y, g, lr):.3f}")
        print(f"dataset acc   : {_cv_auc(E, df['dataset_name'].to_numpy(), g, lr, multiclass=True):.3f}")
        print(f"camera  acc   : {_cv_auc(E, df['domain'].to_numpy(), g, lr, multiclass=True):.3f}")
        for ds, idx in df.groupby("dataset_name").groups.items():
            idx = np.array(list(idx))
            print(f"label AUROC within {ds:10s}: {_cv_auc(E[idx], y[idx], g[idx], lr):.3f}")
        print("Note: embeddings were trained on part of these images, so label AUROC is optimistic; "
              "the point is the dataset/camera accuracy and the within-dataset numbers.")


if __name__ == "__main__":
    main()
