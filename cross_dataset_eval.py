"""
cross_dataset_eval.py -- STEP 4 (the honest generalisation test).

Train on ONE dataset, test on the OTHER (never seen, different camera/site):

    python cross_dataset_eval.py                       # chaksu->papila and papila->chaksu
    python cross_dataset_eval.py --pairs chaksu:papila --epochs 15

Caveat printed with the results: Chaksu's positives are all "suspect" (no
confirmed glaucoma), PAPILA's mix glaucoma + suspect, so label semantics are
not identical across datasets; treat AUROC as the headline number and threshold
metrics as a measure of prior/calibration shift.
"""
import argparse
import os

import pandas as pd

import config
import utils
from dataset import load_index
from engine import predict_df, run_training
from metrics import cluster_bootstrap_ci, safe_auroc, summarize, youden_threshold
from splits import split_patients


def run_pair(df, src: str, dst: str, epochs):
    a = df[df["dataset_name"] == src].copy()
    b = df[df["dataset_name"] == dst].copy()
    mapping = split_patients(a, (0.85, 0.15, 0.0), config.RANDOM_SEED)
    a["split"] = a["patient_id"].map(mapping)
    out_dir = os.path.join(config.CHECKPOINT_DIR, f"lodo_{src}_to_{dst}")
    res = run_training(a[a["split"] == "train"], a[a["split"] == "val"], out_dir, epochs, tag=f"{src}->{dst}")
    p = predict_df(res["model"], b, res["device"])
    thr_src = res["threshold"]
    s = summarize(p["y"], p["prob_pos"], thr_src)
    ci = cluster_bootstrap_ci(p["y"], p["prob_pos"], p["patient_id"], safe_auroc)
    thr_oracle = youden_threshold(p["y"], p["prob_pos"])
    so = summarize(p["y"], p["prob_pos"], thr_oracle)
    print(f"\n[{src} -> {dst}] AUROC {s['auroc']:.3f} "
          f"{'[%.3f, %.3f]' % ci if ci else ''} | @src-val threshold {thr_src:.2f}: sens {s['sensitivity']:.3f} "
          f"spec {s['specificity']:.3f} | @oracle threshold {thr_oracle:.2f}: sens {so['sensitivity']:.3f} spec {so['specificity']:.3f}")
    p.to_csv(os.path.join(config.OUTPUT_DIR, f"lodo_predictions_{src}_to_{dst}.csv"), index=False)
    return {"train_on": src, "test_on": dst, "n_test": len(p), "auroc": s["auroc"],
            "auroc_ci_lo": ci[0] if ci else None, "auroc_ci_hi": ci[1] if ci else None,
            "auprc": s["auprc"], "sens@src_thr": s["sensitivity"], "spec@src_thr": s["specificity"],
            "src_threshold": thr_src, "oracle_threshold": thr_oracle,
            "sens@oracle": so["sensitivity"], "spec@oracle": so["specificity"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=["chaksu:papila", "papila:chaksu"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    utils.print_environment_info(utils.get_device())
    df = load_index()
    rows = [run_pair(df, *pr.split(":"), args.epochs) for pr in args.pairs]
    out = os.path.join(config.OUTPUT_DIR, "lodo_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\nSaved: {out}\nCaveat: Chaksu positives = 'suspect' only; PAPILA = glaucoma + suspect.")


if __name__ == "__main__":
    main()
