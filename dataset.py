"""dataset.py -- index loading + torch Dataset."""
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def load_index(path: str = None) -> pd.DataFrame:
    path = path or config.MERGED_INDEX_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python build_merged_dataset.py --cache` first.")
    df = pd.read_csv(path)
    missing = [c for c in ("image_path", "unified_class", "patient_id", "split", "dataset_name", "domain") if c not in df]
    if missing:
        raise ValueError(f"Index is missing columns {missing}; rebuild it.")
    return df


def build_class_mapping() -> Dict[str, int]:
    return {n: i for i, n in enumerate(config.CLASS_NAMES)}


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], preprocessor, augmentor=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx, self.pre, self.aug = class_to_idx, preprocessor, augmentor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.pre.load_processed(row["image_path"])
        if img is None:
            print(f"[WARN] unreadable image, using black placeholder: {row['image_path']}")
            img = np.zeros((self.pre.image_size, self.pre.image_size, 3), np.uint8)
        if self.aug is not None:
            img = self.aug(img)
        arr = np.ascontiguousarray(self.pre.normalize(img).transpose(2, 0, 1))
        meta = {"dataset_name": row["dataset_name"], "domain": row["domain"],
                "patient_id": row["patient_id"], "image_path": row["image_path"]}
        return torch.from_numpy(arr), self.class_to_idx[row["unified_class"]], meta
