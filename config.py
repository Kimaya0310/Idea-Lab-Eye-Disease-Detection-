"""
config.py -- single source of truth (glaucoma-first pipeline).

Target: binary "Normal" vs "Glaucoma_or_Suspect" (referable glaucoma screening)
trained on NON-MYDRIATIC datasets only: Chaksu + PAPILA.
GRAPE / HRF / DiaRetDB1 are intentionally NOT active (see README, "Why").
"""
import glob
import os
from typing import Dict, List

# --------------------------------------------------------------------------
# 1. PATHS  (override with env var FUNDUS_DATASET_ROOT instead of editing)
# --------------------------------------------------------------------------
DATASET_ROOT = os.environ.get("FUNDUS_DATASET_ROOT", r"d:\Eye Disease Project\Dataset")

# Chaksu layout (from your audit):
#   <root>/Train/Train/1.0_Original_Fundus_Images/{Bosch,Forus,Remidio}/*.JPG
#   <root>/Train/Train/6.0_Glaucoma_Decision/*majority*.csv     (same under Test/Test)
CHAKSU_TRAIN_DIR = os.environ.get("CHAKSU_TRAIN_DIR", os.path.join(DATASET_ROOT, "Train", "Train"))
CHAKSU_TEST_DIR = os.environ.get("CHAKSU_TEST_DIR", os.path.join(DATASET_ROOT, "Test", "Test"))
CHAKSU_IMAGE_SUBDIR = "1.0_Original_Fundus_Images"
CHAKSU_LABEL_SUBDIR = "6.0_Glaucoma_Decision"


def _find_papila(root: str) -> str:
    hits = sorted(glob.glob(os.path.join(root, "PapilaDB*")))
    return hits[0] if hits else os.path.join(root, "PapilaDB")


PAPILA_DIR = os.environ.get("PAPILA_DIR", _find_papila(DATASET_ROOT))
PAPILA_CLINICAL_SUBDIR = "ClinicalData"      # patient_data_od.xlsx / patient_data_os.xlsx
PAPILA_IMAGE_SUBDIR = "FundusImages"         # RET001OD.jpg ...

# Optional, NOT active by default (all-glaucoma dataset -> dataset shortcut).
GRAPE_DIR = os.environ.get("GRAPE_DIR", os.path.join(DATASET_ROOT, "GRAPE_CFPs"))

ACTIVE_DATASETS: List[str] = ["chaksu", "papila"]

# --------------------------------------------------------------------------
# 2. TARGET DEFINITION
# --------------------------------------------------------------------------
NEGATIVE_CLASS = "Normal"
POSITIVE_CLASS = "Glaucoma_or_Suspect"
CLASS_NAMES: List[str] = [NEGATIVE_CLASS, POSITIVE_CLASS]
NUM_CLASSES = 2
# "merge": suspects count as positive (referable). "drop": remove suspects
# (leaves Chaksu with almost no positives -- only for ablation on PAPILA).
SUSPECT_POLICY = "merge"

# --------------------------------------------------------------------------
# 3. SPLITS
# --------------------------------------------------------------------------
RANDOM_SEED = 42
# Datasets with an official split (Chaksu) keep it: official test = our test;
# a patient-grouped VAL_FRAC_OF_TRAIN is carved out of official train for val.
VAL_FRAC_OF_TRAIN = 0.15
# Datasets without an official split (PAPILA): patient-level split.
NO_OFFICIAL_SPLIT_FRACS = (0.70, 0.15, 0.15)

# --------------------------------------------------------------------------
# 4. PREPROCESSING (deterministic, cached to disk)
# --------------------------------------------------------------------------
IMAGE_SIZE = 384                 # disc/cup detail needs more than 224
FOV_THRESHOLD = 12               # gray level (after blur) separating FOV from black border
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
DISC_CROP = True                 # crop a window around the optic disc from the ORIGINAL image (disc_crop.py)
DISC_ROI_FRAC = 0.40             # window side as fraction of the FOV-cropped square (try 0.30-0.50)
COLOR_NORM = "none"              # "none" | "gray_world"  (ablation switch)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
USE_CACHE = True
CACHE_DIR = os.environ.get("FUNDUS_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))

# --------------------------------------------------------------------------
# 5. AUGMENTATION (train only)
# --------------------------------------------------------------------------
AUG_HFLIP_PROB = 0.5
AUG_ROTATION_DEGREES = 15
AUG_SCALE_RANGE = (0.92, 1.08)
AUG_TRANSLATE = 0.04
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.20
AUG_SATURATION = 0.15
AUG_GAMMA = 0.15
AUG_HUE = 0.02
AUG_BLUR_PROB = 0.15
AUG_RESOLUTION_JITTER_PROB = 0.25   # down/up-scale: blunts per-camera sharpness fingerprints

# --------------------------------------------------------------------------
# 6. MODEL / TRAINING
# --------------------------------------------------------------------------
# Baseline first. For the fusion ablation use:
#   ["efficientnet_b0", "densenet121", "resnet50"]
BACKBONES: List[str] = ["efficientnet_b0"]
FUSION_PROJECTION_DIM = 512
DROPOUT = 0.4
FREEZE_BACKBONE_BN = True        # keep pretrained BN statistics fixed (tiny batches)
UNFREEZE_LAST_N_TENSORS = 40     # None = unfreeze everything after FREEZE_EPOCHS

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 30
FREEZE_EPOCHS = 3
WARMUP_EPOCHS = 1
LEARNING_RATE = 3e-4             # head
BACKBONE_LR = 3e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 2
USE_AMP = True
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_METRIC = "val_auroc"   # higher is better

# "dataset_class_sampler": inside EACH dataset, classes are sampled 50/50 while
#   datasets keep their natural share -> the model cannot profit from
#   "Chaksu is mostly Normal, PAPILA has more positives".  (default)
# "class_weighted_ce" | "weighted_sampler" | "focal_loss": classic options.
IMBALANCE_STRATEGY = "dataset_class_sampler"
FOCAL_LOSS_GAMMA = 2.0
LABEL_SMOOTHING = 0.05           # labels are majority votes / "suspect" -> soften hard targets (0 = off)

# EMA of weights: ~340 positives make raw epoch weights noisy; validate/checkpoint the EMA once warm.
USE_EMA = True
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 100

# Test-time augmentation: average each image with its horizontal flip (eval/inference only).
USE_TTA = True

# Offline oversampling (oversample.py): physically written augmented copies of minority-class
# TRAIN images (inherit the source patient_id; never touch val/test). Run
#   python build_merged_dataset.py --cache --oversample
OVERSAMPLE_TARGET_RATIO = 0.6        # minority -> 60% of the domain's majority count
OVERSAMPLE_MAX_COPIES_PER_IMAGE = 4  # cap per source image (keeps copies spread out)

# --------------------------------------------------------------------------
# 7. EVALUATION
# --------------------------------------------------------------------------
BOOTSTRAP_ITERS = 1000
SPECIFICITY_TARGETS = (0.90, 0.95)
SENSITIVITY_TARGETS = (0.90, 0.95)

# --------------------------------------------------------------------------
# 8. OUTPUT PATHS
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MERGED_INDEX_PATH = os.path.join(OUTPUT_DIR, "merged_dataset_index.csv")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")
THRESHOLD_PATH = os.path.join(CHECKPOINT_DIR, "operating_threshold.json")
CLASS_MAPPING_PATH = os.path.join(CHECKPOINT_DIR, "class_mapping.json")
TRAINING_HISTORY_PATH = os.path.join(LOG_DIR, "training_history.csv")
FULL_CONFIG_DUMP_PATH = os.path.join(LOG_DIR, "run_config.json")

for _d in (CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)


def as_dict() -> Dict:
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
