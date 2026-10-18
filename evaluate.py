"""
evaluate.py -- STEP 3.   python evaluate.py [--split test|val] [--checkpoint PATH]

Screening-oriented report with patient-clustered bootstrap CIs:
overall + per dataset + per camera, sensitivity@fixed-specificity, calibration.
The operating threshold comes from VALIDATION (operating_threshold.json).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

import config
import utils
from dataset import load_index
from engine import build_model_from_checkpoint, predict_df
from metrics import (cluster_bootstrap_ci, confusion_at, safe_auprc, safe_auroc, summarize)


def _fmt(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "[n/a]"


def report_block(name, d, thr, iters):
    s = summarize(d["y"], d["prob_pos"], thr)
    ci_auc = ci_ap = None
    if d["y"].nunique() == 2 and len(d) >= 30:
        ci_auc = cluster_bootstrap_ci(d["y"], d["prob_pos"], d["patient_id"], safe_auroc, iters)
        ci_ap = cluster_bootstrap_ci(d["y"], d["prob_pos"], d["patient_id"], safe_auprc, iters)
    s.update({"name": name, "auroc_ci_lo": ci_auc[0] if ci_auc else np.nan, "auroc_ci_hi": ci_auc[1] if ci_auc else np.nan,
              "auprc_ci_lo": ci_ap[0] if ci_ap else np.nan, "auprc_ci_hi": ci_ap[1] if ci_ap else np.nan})
    print(f"{name:22s} n={s['n']:4d} prev={s['prevalence']:.2f} | AUROC {s['auroc']:.3f} {_fmt(ci_auc)} | "
          f"AUPRC {s['auprc']:.3f} {_fmt(ci_ap)} | sens {s['sensitivity']:.3f} spec {s['specificity']:.3f} "
          f"@thr {thr:.2f}" + (f" | sens@spec90 {s.get('sens@spec90', float('nan')):.3f}" if 'sens@spec90' in s else ""))
    return s


def plots(d, thr, out_dir, title):
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for name, g in [("ALL", d)] + [(k, v) for k, v in d.groupby("dataset_name")]:
        if g["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["y"], g["prob_pos"]); ax[0].plot(fpr, tpr, label=f"{name} ({safe_auroc(g['y'], g['prob_pos']):.3f})")
        pr, rc, _ = precision_recall_curve(g["y"], g["prob_pos"]); ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.3); ax[0].set(title="ROC", xlabel="1-specificity", ylabel="sensitivity"); ax[0].legend()
    ax[1].set(title="Precision-Recall", xlabel="recall", ylabel="precision"); ax[1].legend()
    bins = np.linspace(0, 1, 11); idx = np.digitize(d["prob_pos"], bins[1:-1])
    xs = [d["prob_pos"][idx == i].mean() for i in range(10) if (idx == i).any()]
    ys = [d["y"][idx == i].mean() for i in range(10) if (idx == i).any()]
    ax[2].plot([0, 1], [0, 1], "k--", alpha=.3); ax[2].plot(xs, ys, "o-"); ax[2].set(title="Reliability", xlabel="predicted P(pos)", ylabel="observed rate")
    c = confusion_at(d["y"], d["prob_pos"], thr)
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    ax[3].imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax[3].text(j, i, int(v), ha="center", va="center", fontsize=14)
    ax[3].set(title=f"Confusion @ thr {thr:.2f}", xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Pos"], yticklabels=["Normal", "Pos"])
    fig.suptitle(title); plt.tight_layout()
    path = os.path.join(out_dir, "evaluation_plots.png"); plt.savefig(path, dpi=140); plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    ap.add_argument("--iters", type=int, default=config.BOOTSTRAP_ITERS)
    args = ap.parse_args()

    device = utils.get_device()
    model, ck = build_model_from_checkpoint(args.checkpoint, device)
    ck_dir = os.path.dirname(args.checkpoint)
    thr = utils.load_json(os.path.join(ck_dir, "operating_threshold.json"))["threshold"]

    df = load_index()
    df = df[df["dataset_name"].isin(config.ACTIVE_DATASETS) & (df["split"] == args.split)]
    d = predict_df(model, df, device)
    print(f"\nModel {ck['backbone_names']} epoch {ck['epoch']} | {args.split} n={len(d)} | threshold from val = {thr:.3f}\n")

    rows = [report_block("ALL", d, thr, args.iters)]
    for ds, g in d.groupby("dataset_name"):
        rows.append(report_block(ds, g, thr, args.iters))
    print("\n-- by camera / domain --")
    for dom, g in d.groupby("domain"):
        rows.append(report_block(dom, g, thr, args.iters))
    res = pd.DataFrame(rows)
    out = os.path.join(config.OUTPUT_DIR, f"metrics_{args.split}.csv"); res.to_csv(out, index=False)
    d.to_csv(os.path.join(config.OUTPUT_DIR, f"predictions_{args.split}.csv"), index=False)
    plots(d, thr, config.OUTPUT_DIR, f"{args.split} | {ck['backbone_names']}")
    print(f"Saved: {out}")
    print("\nHow to read this: the ALL row can look good purely from dataset differences. "
          "Trust the per-dataset rows (both classes exist inside each) and cross_dataset_eval.py.")


if __name__ == "__main__":
    main()
