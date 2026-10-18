"""
ensemble.py -- train K seeds, average probabilities, report on the same test split.

    python ensemble.py --seeds 1 2 3 4 5            # train + evaluate
    python ensemble.py --seeds 1 2 3 4 5 --eval-only

Why: with ~340 positives the model-to-model variance is large (your test AUROC
CI is [0.67, 0.80]). Averaging independently seeded models is the most reliable
way to raise AUROC/AUPRC on a small dataset. Threshold is chosen on the
AVERAGED validation probabilities only (never on test).
"""
import argparse, os
import numpy as np, pandas as pd
import config, utils
from dataset import load_index
from engine import build_model_from_checkpoint, predict_df, run_training
from metrics import youden_threshold
from evaluate import report_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--iters", type=int, default=config.BOOTSTRAP_ITERS)
    a = ap.parse_args()

    device = utils.get_device()
    df = load_index()
    df = df[df["dataset_name"].isin(config.ACTIVE_DATASETS)]
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))

    val_p, test_p = [], []
    for sd in a.seeds:
        out = os.path.join(config.CHECKPOINT_DIR, f"ens_seed{sd}")
        if not a.eval_only:
            config.RANDOM_SEED = sd                  # run_training reads this at call time
            run_training(tr, va, out, tag=f"seed{sd}")
        model, _ = build_model_from_checkpoint(os.path.join(out, "best_model.pth"), device)
        val_p.append(predict_df(model, va, device)["prob_pos"].to_numpy())
        t = predict_df(model, te, device)
        test_p.append(t["prob_pos"].to_numpy())
        del model

    vy = va["unified_class"].map({n: i for i, n in enumerate(config.CLASS_NAMES)}).to_numpy()
    thr = youden_threshold(vy, np.mean(val_p, 0))
    d = t.copy(); d["prob_pos"] = np.mean(test_p, 0)
    print(f"\nEnsemble of {len(a.seeds)} models | threshold (val-averaged) = {thr:.3f}\n")
    rows = [report_block("ALL", d, thr, a.iters)]
    for ds, g in d.groupby("dataset_name"):
        rows.append(report_block(ds, g, thr, a.iters))
    for dom, g in d.groupby("domain"):
        rows.append(report_block(dom, g, thr, a.iters))
    pd.DataFrame(rows).to_csv(os.path.join(config.OUTPUT_DIR, "metrics_test_ensemble.csv"), index=False)
    d.to_csv(os.path.join(config.OUTPUT_DIR, "predictions_test_ensemble.csv"), index=False)


if __name__ == "__main__":
    main()
