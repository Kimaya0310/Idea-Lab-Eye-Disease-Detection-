# Glaucoma-first fundus classifier (non-mydriatic)

Binary screening model: **Normal vs Glaucoma_or_Suspect**, trained on the two
datasets that are verified non-mydriatic and have both classes: **Chaksu** and **PAPILA**.

## Run order
```bash
pip install -r requirements.txt
set FUNDUS_DATASET_ROOT=d:\Eye Disease Project\Dataset      # or edit config.py

python build_merged_dataset.py --cache   # 1. index + splits + preprocessed cache  (paste me this log)
python train.py                          # 2. baseline: EfficientNet-B0 @ 384px
python evaluate.py                       # 3. test metrics with patient-clustered CIs
python cross_dataset_eval.py             # 4. train on one dataset, test on the other
python dataset_probe.py [--embeddings]   # 5. shortcut diagnostics
python inference.py some_image.jpg       # single image + Grad-CAM
```
On Windows, if the DataLoader errors, set `NUM_WORKERS = 0` in `config.py`.

## Why this design
| Issue found in the earlier code | What this version does |
|---|---|
| DR from 15 HRF images; 2-3 images in val/test | DR removed from this stage (add back when you have non-mydriatic DR data) |
| GRAPE = 64% of positives, zero Normals -> dataset shortcut | GRAPE not active (loader kept, regex fixed, in `dataset_sources.py`) |
| Suspects merged blindly, 3 classes | Explicit binary "referable" target; `SUSPECT_POLICY` switch |
| Chaksu: recomputed vote, own random split, one folder glob | Official majority CSVs, recursive layout, camera-aware matching, **official Train/Test kept** |
| Dataset positive rate leaks the label (Chaksu ~14%, PAPILA ~32%) | `dataset_class_sampler`: classes 50/50 inside each dataset |
| 224px squashed non-square crops | FOV crop -> pad to square -> 384px; cached to disk |
| Frozen backbones still updated BN stats; scheduler + optimizer reset at unfreeze | BN kept fixed; optimizer built once; single warmup+cosine schedule |
| Model selection on 2-image val F1 | Selection on val AUROC; threshold chosen on val only |
| Split recomputed in every script | Split saved once in `merged_dataset_index.csv` |
| Grad-CAM full-backward hook (breaks on DenseNet in-place ReLU) | forward hook + tensor hook |
| Macro-F1 on single-class datasets | AUROC/AUPRC/sens@spec with patient-clustered bootstrap CIs |

## Splits
* **Chaksu**: official Train -> train (85%) / val (15%, grouped); official Test -> test.
* **PAPILA**: patient-level 70/15/15 (both eyes of a patient always together).
* Chaksu has **no patient IDs**; each image is its own group. If both eyes of one
  patient sit in train and test, results are slightly optimistic. Ask the dataset authors
  or check filenames if you can recover patient IDs.

## What to expect / how to read results
* The **ALL** row mixes datasets and can look good for the wrong reason. Trust the
  **per-dataset rows** and **cross_dataset_eval.py** (cross-dataset AUROC is the number
  that predicts real-world behaviour). Expect it to be lower than in-distribution.
* Chaksu positives are all "suspect"; PAPILA mixes glaucoma + suspect. Labels are not
  semantically identical across datasets.
* `dataset_probe.py`: if trivial image statistics predict the label pooled but not within a
  dataset, the label is entangled with the dataset (expected here; the sampler mitigates it).

## Ablations (one change at a time, compare cross-dataset AUROC)
1. `BACKBONES = ["efficientnet_b0","densenet121","resnet50"]` (fusion) vs baseline
2. `IMAGE_SIZE` 384 vs 512 (delete/keep `cache/`; cache is keyed by settings)
3. `ENABLE_CLAHE` on/off, `COLOR_NORM = "gray_world"`
4. `IMBALANCE_STRATEGY = "class_weighted_ce"` vs `dataset_class_sampler`
5. Add GRAPE as *pre-training only* (never as a labelled class)

## Verification status
Tested here on mock data laid out like your folders: index building (Chaksu + PAPILA),
splits + leakage assert, preprocessing cache, sampler weights, metadata probe.
**Not executed here (no torch install available):** `engine.py`, `train.py`, `evaluate.py`,
`cross_dataset_eval.py`, `inference.py`, `explainability.py` -- they pass static checks
(pyflakes), but expect to fix small issues on the first real run. Chaksu CSV column
names were inferred from your audit; the loader prints the columns it picked.

*Research baseline. Not a medical device; not clinically validated.*


## Performance round (after metrics_test.csv / lodo_summary.csv)
Baseline: test AUROC 0.738 [0.67-0.80]; cross-dataset 0.63-0.66; sensitivity at the val threshold only 0.39 (Chaksu 0.23).
Changes (all switchable in config.py):
* `DISC_CROP` / `DISC_ROI_FRAC` (disc_crop.py): crop a window around the optic disc from the ORIGINAL image. New cache is built (tag includes it).
  Heuristic, not a trained detector -- ~11/12 correct on a visual check; it can miss very dark images.
* `LABEL_SMOOTHING`, `USE_EMA` (EMA weights validated/checkpointed after warm-up), `USE_TTA` (flip-average at eval/inference).
* `python build_merged_dataset.py --cache --oversample`: offline augmented copies of minority-class TRAIN images (oversample.py).
  Copies inherit patient_id, never touch val/test; `is_synthetic` column marks them.
* `python ensemble.py --seeds 1 2 3 4 5`: multi-seed averaging, threshold from averaged val -> outputs/metrics_test_ensemble.csv.
Not executed here (no torch/GPU): engine.py, ensemble.py. Compare each change one at a time on cross-dataset AUROC.
