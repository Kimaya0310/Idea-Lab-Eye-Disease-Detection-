"""
splits.py -- leakage-safe splits (no torch dependency).

* Datasets with an official split (Chaksu) keep it: official test -> "test";
  a patient-grouped slice of official train -> "val".
* Datasets without one (PAPILA) get a patient-level 70/15/15 split, stratified
  by each patient's class. Both eyes of a PAPILA patient stay together.
"""
import numpy as np
import pandas as pd

import config


def split_patients(df: pd.DataFrame, fracs, seed: int) -> dict:
    train_f, val_f, test_f = fracs
    assert abs(train_f + val_f + test_f - 1.0) < 1e-6
    rep = df.groupby("patient_id")["unified_class"].agg(lambda s: s.value_counts().idxmax())
    rng = np.random.RandomState(seed)
    out = {}
    for cls in sorted(rep.unique()):
        pats = np.array(sorted(rep.index[rep == cls]), dtype=object)
        rng.shuffle(pats)
        n = len(pats)
        if n < 3:
            print(f"[WARN] class '{cls}' has {n} patient(s) in this group -> all to train")
            out.update({p: "train" for p in pats})
            continue
        n_test = int(round(n * test_f)) if test_f > 0 else 0
        n_val = max(1, int(round(n * val_f))) if val_f > 0 else 0
        n_train = n - n_val - n_test
        for p in pats[:n_train]:
            out[p] = "train"
        for p in pats[n_train:n_train + n_val]:
            out[p] = "val"
        for p in pats[n_train + n_val:]:
            out[p] = "test"
    return out


def assign_splits(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    df = df.copy()
    df["split"] = None
    for name, sub in df.groupby("dataset_name"):
        has_official = sub["official_split"].notna().all()
        if has_official:
            test_idx = sub.index[sub["official_split"] == "test"]
            pool = sub[sub["official_split"] == "train"]
            df.loc[test_idx, "split"] = "test"
            if len(pool):
                mapping = split_patients(pool, (1 - config.VAL_FRAC_OF_TRAIN, config.VAL_FRAC_OF_TRAIN, 0.0), seed)
                df.loc[pool.index, "split"] = pool["patient_id"].map(mapping)
        else:
            mapping = split_patients(sub, config.NO_OFFICIAL_SPLIT_FRACS, seed)
            df.loc[sub.index, "split"] = sub["patient_id"].map(mapping)
    df["split"] = df["split"].fillna("train")
    leak = df.groupby("patient_id")["split"].nunique()
    if (leak > 1).any():
        raise RuntimeError(f"Patient leakage: {list(leak[leak > 1].index[:10])}")
    return df


def report_splits(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70 + "\nSPLIT SUMMARY (images)\n" + "=" * 70)
    print(pd.crosstab([df["domain"], df["unified_class"]], df["split"], margins=True).to_string())
    print("\nPatients per split:")
    print(df.groupby("split")["patient_id"].nunique().to_string())
    print("=" * 70)
