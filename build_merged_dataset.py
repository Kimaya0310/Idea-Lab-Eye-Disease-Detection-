"""
build_merged_dataset.py -- STEP 1 (no torch, no GPU).

    python build_merged_dataset.py            # index + splits -> outputs/merged_dataset_index.csv
    python build_merged_dataset.py --cache    # also pre-computes the preprocessed image cache
                                              # and drops unreadable images from the index

train.py / evaluate.py / cross_dataset_eval.py all READ this CSV, so every
script sees exactly the same split.
"""
import argparse
from multiprocessing import Pool

import pandas as pd

import config
from dataset_sources import build_unified_index
from preprocessing import FundusPreprocessor
from splits import assign_splits, report_splits


def _cache_one(path: str) -> bool:
    return FundusPreprocessor().load_processed(path) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="precompute preprocessed images")
    ap.add_argument("--oversample", action="store_true",
                    help="add augmented copies of minority-class TRAIN images (oversample.py); implies --cache")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.cache = args.cache or args.oversample

    df = build_unified_index()

    if args.cache:
        print(f"\nCaching {len(df)} images -> {FundusPreprocessor().cache_dir}")
        with Pool(args.workers) as pool:
            ok = pool.map(_cache_one, df["image_path"].tolist(), chunksize=8)
        bad = df[~pd.Series(ok, index=df.index)]
        if len(bad):
            print(f"[WARN] {len(bad)} unreadable images dropped, e.g. {bad['image_path'].head(5).tolist()}")
            df = df[pd.Series(ok, index=df.index)].reset_index(drop=True)

    df = assign_splits(df)
    report_splits(df)
    if args.oversample:
        from oversample import oversample_train_split
        df = oversample_train_split(df)
    df.to_csv(config.MERGED_INDEX_PATH, index=False)

    print(f"\nSaved: {config.MERGED_INDEX_PATH}  ({len(df)} images)")
    print("\nCLASS x ORIGINAL LABEL:")
    print(pd.crosstab(df["domain"], [df["unified_class"], df["original_class"]]).to_string())
    print("\nSanity checks to eyeball:")
    print(" - Chaksu should total ~1345 (train 1009 / test 336); PAPILA 488 (244 patients).")
    print(" - Positive rate differs by dataset (Chaksu ~14%, PAPILA ~32%): this is exactly why "
          "training uses domain x class balanced sampling.")
    print(" - Next: python train.py")


if __name__ == "__main__":
    main()
