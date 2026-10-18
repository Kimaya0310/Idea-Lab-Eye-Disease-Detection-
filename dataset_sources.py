"""
dataset_sources.py -- dataset-specific loaders (no torch dependency).

Every loader returns the same schema:
    image_path, dataset_name, domain, original_class, unified_class,
    patient_id, eye_id, official_split

`domain` = dataset (+ camera for Chaksu). It is used for balanced sampling,
per-domain reporting and shortcut probes.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SCHEMA = ["image_path", "dataset_name", "domain", "original_class",
          "unified_class", "patient_id", "eye_id", "official_split"]
CAMERAS = ("bosch", "forus", "remidio")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def to_unified(original_class: str) -> Optional[str]:
    """original_class in {Normal, Glaucoma, Glaucoma_Suspect} -> binary target."""
    if original_class == "Normal":
        return config.NEGATIVE_CLASS
    if original_class == "Glaucoma":
        return config.POSITIVE_CLASS
    if original_class == "Glaucoma_Suspect":
        return config.POSITIVE_CLASS if config.SUSPECT_POLICY == "merge" else None
    return None


def _camera_in(text: str) -> Optional[str]:
    t = text.lower()
    for c in CAMERAS:
        if c in t:
            return c
    return None


def _walk_images(root: str):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                yield os.path.join(dirpath, f)


# ============================================================================
# CHAKSU -- official majority-decision CSVs + official Train/Test split
# ============================================================================
def _norm_vote(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    t = re.sub(r"[_\-\s]+", " ", str(raw).strip().lower())
    if "suspect" in t:
        return "Glaucoma_Suspect"
    if "glaucoma" in t:
        return "Glaucoma"
    if "normal" in t:
        return "Normal"
    return None


def _chaksu_key_candidates(cell) -> List[str]:
    name = Path(str(cell).replace("\\", "/")).name.strip()
    cands = []
    # Split only on '-' since '_' is part of valid image stems (e.g., IMG_3709, P17_Image1)
    tokens = [name]
    if "-" in name:
        tokens.extend(name.split("-"))
    for token in tokens:
        clean = re.sub(r"\.(jpe?g|png|tiff?|bmp)$", "", token.strip(), flags=re.I).rstrip(". ").strip().lower()
        if clean:
            cands.append(clean)
    return list(dict.fromkeys(cands))


def _pick_column(columns, must_contain: List[str], prefer: List[str] = ()) -> Optional[str]:
    low = {c: str(c).strip().lower() for c in columns}
    for p in prefer:
        for c, l in low.items():
            if l == p:
                return c
    for c, l in low.items():
        if all(m in l for m in must_contain):
            return c
    return None


def _load_chaksu_split(split_name: str, split_dir: str) -> Tuple[List[dict], Dict[str, int]]:
    stats = {"csv_files": 0, "rows": 0, "unmatched": 0, "ambiguous": 0, "no_label": 0}
    if not os.path.isdir(split_dir):
        print(f"[WARN] Chaksu {split_name} dir not found: {split_dir}")
        return [], stats

    img_root = os.path.join(split_dir, config.CHAKSU_IMAGE_SUBDIR)
    img_root = img_root if os.path.isdir(img_root) else split_dir
    by_cam_stem: Dict[Tuple[Optional[str], str], str] = {}
    by_stem: Dict[str, List[str]] = {}
    for p in _walk_images(img_root):
        stem = os.path.splitext(os.path.basename(p))[0].strip().lower()
        by_cam_stem[(_camera_in(p), stem)] = p
        by_stem.setdefault(stem, []).append(p)

    label_root = os.path.join(split_dir, config.CHAKSU_LABEL_SUBDIR)
    label_root = label_root if os.path.isdir(label_root) else split_dir
    csvs = sorted(
        os.path.join(d, f) for d, _, fs in os.walk(label_root) for f in fs
        if f.lower().endswith(".csv") and "majority" in f.lower()
    )
    if not csvs:
        print(f"[WARN] Chaksu {split_name}: no '*majority*.csv' under {label_root}")
    rows, seen = [], set()
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] cannot read {csv_path}: {e}")
            continue
        img_col = _pick_column(df.columns, ["image"])
        lab_col = (_pick_column(df.columns, ["majority"], prefer=["majority decision"])
                   or _pick_column(df.columns, ["decision"], prefer=["glaucoma decision"]))
        print(f"[INFO] Chaksu {split_name} | {os.path.basename(csv_path)} | image col='{img_col}' "
              f"label col='{lab_col}' | {len(df)} rows")
        if img_col is None or lab_col is None:
            print(f"[WARN]   columns were {list(df.columns)} -- file skipped; set names in dataset_sources.")
            continue
        stats["csv_files"] += 1
        csv_cam = _camera_in(os.path.basename(csv_path))
        for _, r in df.iterrows():
            cell = r[img_col]
            if pd.isna(cell):
                continue
            original = _norm_vote(r[lab_col])
            unified = to_unified(original) if original else None
            if unified is None:
                stats["no_label"] += 1
                continue
            cam_hint = _camera_in(str(cell)) or csv_cam
            path, ambiguous = None, False
            for cand in _chaksu_key_candidates(cell):
                if cam_hint and (cam_hint, cand) in by_cam_stem:
                    path = by_cam_stem[(cam_hint, cand)]
                    break
                lst = by_stem.get(cand, [])
                if len(lst) == 1:
                    path = lst[0]
                    break
                if len(lst) > 1:
                    ambiguous = True
            if path is None:
                stats["ambiguous" if ambiguous else "unmatched"] += 1
                continue
            if path in seen:
                continue
            seen.add(path)
            cam = _camera_in(path) or "unknown"
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            rows.append({
                "image_path": path, "dataset_name": "chaksu", "domain": f"chaksu_{cam}",
                "original_class": original, "unified_class": unified,
                # Chaksu ships no patient IDs -> one image = one group (see README caveat);
                # the official split is part of the ID so same-named files in Train/Test never merge
                "patient_id": f"chaksu_{split_name}_{cam}_{stem}", "eye_id": None,
                "official_split": split_name,
            })
            stats["rows"] += 1
    return rows, stats


def load_chaksu() -> pd.DataFrame:
    all_rows = []
    for split_name, split_dir in (("train", config.CHAKSU_TRAIN_DIR), ("test", config.CHAKSU_TEST_DIR)):
        rows, st = _load_chaksu_split(split_name, split_dir)
        print(f"[INFO] Chaksu {split_name}: kept {st['rows']} | unmatched images {st['unmatched']} | "
              f"ambiguous filenames {st['ambiguous']} | unusable labels {st['no_label']}")
        all_rows += rows
    return pd.DataFrame(all_rows, columns=SCHEMA) if all_rows else _empty()


# ============================================================================
# PAPILA -- clinical Excel (per eye), Diagnosis: 0 healthy / 1 glaucoma / 2 suspect
# ============================================================================
_PAPILA_DIAG = {0: "Normal", 1: "Glaucoma", 2: "Glaucoma_Suspect"}


def _read_papila_sheet(path: str):
    for header in (1, 0, 2):
        try:
            df = pd.read_excel(path, header=header)
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        id_col = cols.get("id") or next((c for l, c in cols.items() if l.startswith("id") or "patient" in l), None)
        dg_col = next((c for l, c in cols.items() if "diagnos" in l), None)
        if id_col is not None and dg_col is not None:
            return df, id_col, dg_col, header

    # Fallback for sheets with multi-row headers (e.g. ID in row 2, Diagnosis in row 0/1)
    try:
        df_raw = pd.read_excel(path, header=None)
        id_col_idx, dg_col_idx = None, None
        for r in range(min(5, len(df_raw))):
            for c in range(df_raw.shape[1]):
                val = str(df_raw.iat[r, c]).strip().lower()
                if id_col_idx is None and (val == "id" or "patient" in val):
                    id_col_idx = c
                if dg_col_idx is None and "diagnos" in val:
                    dg_col_idx = c
        if id_col_idx is not None and dg_col_idx is not None:
            start_row = 3
            for r in range(len(df_raw)):
                v = str(df_raw.iat[r, id_col_idx]).strip()
                if v.startswith("#") or v.isdigit():
                    start_row = r
                    break
            data = df_raw.iloc[start_row:].copy()
            data = data.rename(columns={id_col_idx: "id", dg_col_idx: "diagnosis"})
            return data, "id", "diagnosis", start_row
    except Exception:
        pass

    return None


def load_papila() -> pd.DataFrame:
    clin = os.path.join(config.PAPILA_DIR, config.PAPILA_CLINICAL_SUBDIR)
    imgd = os.path.join(config.PAPILA_DIR, config.PAPILA_IMAGE_SUBDIR)
    if not os.path.isdir(clin) or not os.path.isdir(imgd):
        print(f"[WARN] PAPILA dirs not found ({clin} / {imgd}). Set PAPILA_DIR.")
        return _empty()
    index = {os.path.splitext(os.path.basename(p))[0].strip().lower(): p for p in _walk_images(imgd)}
    rows, unmatched = [], []
    for eye in ("OD", "OS"):
        xlsx = os.path.join(clin, f"patient_data_{eye.lower()}.xlsx")
        if not os.path.exists(xlsx):
            print(f"[WARN] missing {xlsx}")
            continue
        parsed = _read_papila_sheet(xlsx)
        if parsed is None:
            print(f"[WARN] {xlsx}: could not find ID + Diagnosis columns")
            continue
        df, id_col, dg_col, header = parsed
        print(f"[INFO] PAPILA {eye}: header row={header} id col='{id_col}' diagnosis col='{dg_col}' | {len(df)} rows")
        for _, r in df.iterrows():
            digits = re.sub(r"\D", "", str(r[id_col]))
            try:
                code = int(r[dg_col])
            except (TypeError, ValueError):
                continue
            if not digits or code not in _PAPILA_DIAG:
                continue
            original = _PAPILA_DIAG[code]
            unified = to_unified(original)
            if unified is None:
                continue
            pid = int(digits)
            key = f"ret{pid:03d}{eye.lower()}"
            path = index.get(key)
            if path is None:
                unmatched.append(key)
                continue
            rows.append({
                "image_path": path, "dataset_name": "papila", "domain": "papila",
                "original_class": original, "unified_class": unified,
                "patient_id": f"papila_{pid:03d}", "eye_id": eye, "official_split": None,
            })
    if unmatched:
        print(f"[WARN] PAPILA: {len(unmatched)} rows without image, e.g. {unmatched[:5]} "
              f"(available stems e.g. {list(index)[:3]})")
    return pd.DataFrame(rows, columns=SCHEMA) if rows else _empty()


# ============================================================================
# GRAPE -- OPTIONAL, not active by default (all-glaucoma => dataset shortcut)
# ============================================================================
_GRAPE_RE = re.compile(r"^(\d+)[_\-](\d+)[_\-](\d+)")   # patient_eye_visit


def load_grape() -> pd.DataFrame:
    if not os.path.isdir(config.GRAPE_DIR):
        print(f"[WARN] GRAPE dir not found: {config.GRAPE_DIR}")
        return _empty()
    rows = []
    for p in _walk_images(config.GRAPE_DIR):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = _GRAPE_RE.match(stem)
        pid, eye = (m.group(1), m.group(2)) if m else (stem, None)
        rows.append({"image_path": p, "dataset_name": "grape", "domain": "grape",
                     "original_class": "Glaucoma", "unified_class": config.POSITIVE_CLASS,
                     "patient_id": f"grape_{pid}", "eye_id": eye, "official_split": None})
    return pd.DataFrame(rows, columns=SCHEMA) if rows else _empty()


DATASET_LOADERS = {"chaksu": load_chaksu, "papila": load_papila, "grape": load_grape}


def build_unified_index(active: List[str] = None) -> pd.DataFrame:
    frames = []
    for name in (active or config.ACTIVE_DATASETS):
        print(f"\n--- Loading '{name}' ---")
        frames.append(DATASET_LOADERS[name]())
    valid_frames = [f for f in frames if not f.empty]
    df = pd.concat(valid_frames, ignore_index=True) if valid_frames else _empty()
    if df.empty:
        raise RuntimeError("No images indexed. Check paths in config.py / the [WARN] lines above.")
    dup = df.duplicated("image_path", keep=False).sum()
    if dup:
        print(f"[WARN] {dup} rows share an image_path; dropping duplicates.")
        df = df.drop_duplicates("image_path").reset_index(drop=True)
    return df
