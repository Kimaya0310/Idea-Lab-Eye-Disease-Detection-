"""
trim_dataset.py -- shrink the fundus dataset to only what glaucoma_classifier uses.

SAFE BY DESIGN
  * Default is a DRY RUN: nothing is touched, it only prints/saves a plan.
  * With --apply, files are MOVED (never deleted) into a quarantine folder that
    sits next to the dataset, keeping their relative paths.  Same drive => instant
    renames, no extra disk space.  Delete the quarantine folder yourself once
    you have verified the pipeline.
  * KEEP list = (a) every image in used_images_manifest.csv
                (b) Chaksu label CSVs the code actually reads: any *.csv whose
                    FILE NAME contains "majority" anywhere under
                    <Train|Test>/<Train|Test>/6.0_Glaucoma_Decision  (mirrors dataset_sources.py)
                (c) PAPILA ClinicalData/patient_data_od.xlsx and patient_data_os.xlsx
                (d) optional: --keep-grape keeps the CFPs folder.

USAGE (Windows, from any folder):
  python trim_dataset.py --root "D:\\Eye Disease Project\\Dataset" --manifest used_images_manifest.csv
  python trim_dataset.py --root ... --manifest ... --apply
  python trim_dataset.py --root ... --manifest ... --apply --keep-grape
  python trim_dataset.py --root ... --restore          # put everything back
"""
import argparse
import csv
import os
import shutil
import sys

LABEL_SUBDIR = "6.0_Glaucoma_Decision"


def norm(p):
    return os.path.normcase(os.path.normpath(p.replace("\\", os.sep)))


def build_keep(root, manifest, keep_grape):
    keep = set()
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keep.add(norm(row["image_path"]))
    n_img = len(keep)

    # Chaksu label CSVs: mirror dataset_sources.py (recursive, "majority" in file name)
    for split in ("Train", "Test"):
        label_root = os.path.join(root, split, split, LABEL_SUBDIR)
        for d, _, fs in os.walk(label_root):
            for fn in fs:
                if fn.lower().endswith(".csv") and "majority" in fn.lower():
                    keep.add(norm(os.path.join(d, fn)))
    # PAPILA clinical sheets
    for e in os.scandir(root):
        if e.is_dir() and e.name.startswith("PapilaDB"):
            for fn in ("patient_data_od.xlsx", "patient_data_os.xlsx"):
                keep.add(norm(os.path.join(e.path, "ClinicalData", fn)))
    if keep_grape:
        for d, _, fs in os.walk(os.path.join(root, "CFPs")):
            for fn in fs:
                keep.add(norm(os.path.join(d, fn)))
    return keep, n_img


def plan(root, keep, quarantine):
    """Walk top-down. A directory with no kept file inside is moved whole."""
    keep_dirs = set()
    for k in keep:
        p = os.path.dirname(k)
        while len(p) >= len(norm(root)):
            keep_dirs.add(p)
            np_ = os.path.dirname(p)
            if np_ == p:
                break
            p = np_
    moves, kept_found = [], set()
    qn = norm(quarantine)

    def visit(d):
        for e in sorted(os.scandir(d), key=lambda x: x.name):
            n = norm(e.path)
            if n == qn:
                continue
            if e.is_dir(follow_symlinks=False):
                if n in keep_dirs:
                    visit(e.path)
                else:
                    moves.append(e.path)
            elif n in keep:
                kept_found.add(n)
            else:
                moves.append(e.path)

    visit(root)
    return moves, kept_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", help="used_images_manifest.csv")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-grape", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--quarantine", help="default: <root>_QUARANTINE (sibling of root)")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    quarantine = os.path.abspath(a.quarantine) if a.quarantine else root.rstrip("\\/") + "_QUARANTINE"

    if a.restore:
        if not os.path.isdir(quarantine):
            sys.exit(f"nothing to restore: {quarantine} not found")
        n = 0
        log = os.path.join(quarantine, "_moved.csv")
        with open(log, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                src, dst = row["quarantine_path"], row["original_path"]
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
                    n += 1
        print(f"restored {n} items. You can delete {quarantine} now.")
        return

    if not a.manifest:
        sys.exit("--manifest is required")
    keep, n_img = build_keep(root, a.manifest, a.keep_grape)
    moves, kept_found = plan(root, keep, quarantine)

    missing = sorted(k for k in keep if k not in kept_found)
    print(f"Root:        {root}")
    print(f"Quarantine:  {quarantine}")
    print(f"Keep rules:  {len(keep)} files ({n_img} images from manifest)")
    print(f"Found on disk to KEEP: {len(kept_found)}")
    print(f"Top-level items to MOVE: {len(moves)}")
    if missing:
        print(f"\n!! {len(missing)} keep-listed files were NOT found on disk (first 10):")
        for m in missing[:10]:
            print("   ", m)
        if a.apply:
            sys.exit("Refusing to --apply while keep-listed files are missing. "
                     "Check --root / manifest paths (drive letter case is ignored).")

    plan_path = os.path.join(os.getcwd(), "trim_plan.csv")
    with open(plan_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_path", "type"])
        for m in moves:
            w.writerow([m, "dir" if os.path.isdir(m) else "file"])
    print(f"Plan written to {plan_path}")

    if not a.apply:
        print("\nDRY RUN only. Re-run with --apply to move these into quarantine.")
        return

    os.makedirs(quarantine, exist_ok=True)
    log = os.path.join(quarantine, "_moved.csv")
    new = not os.path.exists(log)
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["original_path", "quarantine_path"])
        for i, src in enumerate(moves, 1):
            rel = os.path.relpath(src, root)
            dst = os.path.join(quarantine, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            w.writerow([src, dst])
            if i % 200 == 0:
                print(f"  moved {i}/{len(moves)}")
    print(f"Done. Moved {len(moves)} items to {quarantine}")
    print("Next: run  python build_merged_dataset.py  and confirm 1833 rows, "
          "then delete the quarantine folder when satisfied.")


if __name__ == "__main__":
    main()
