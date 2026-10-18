"""
train.py -- STEP 2.   python train.py

Reads outputs/merged_dataset_index.csv (build it first) so train / evaluate /
LODO all share the identical patient-level split.
"""
import config
import utils
from dataset import load_index, build_class_mapping
from engine import run_training


def main():
    utils.set_seed(config.RANDOM_SEED)
    utils.print_environment_info(utils.get_device())
    utils.save_json(config.as_dict(), config.FULL_CONFIG_DUMP_PATH)
    utils.save_json(build_class_mapping(), config.CLASS_MAPPING_PATH)

    df = load_index()
    df = df[df["dataset_name"].isin(config.ACTIVE_DATASETS)]
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {(df['split'] == 'test').sum()} (untouched until evaluate.py)")

    run_training(train_df, val_df, config.CHECKPOINT_DIR, tag="main")
    print("\nDone. Next: python evaluate.py")


if __name__ == "__main__":
    main()
