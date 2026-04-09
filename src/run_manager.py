import os
import json
from datetime import datetime


def create_run_dir(base_dir, task_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{task_name}_{timestamp}"
    run_dir = os.path.join(base_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_name


def config_group_combination(cfg):
    """Derive (group, combination) folder names from a config dict.

    Structure: run_dir / group / combination
      group      = {model_type}_{aug|noaug}   e.g. scratch_aug, pretrained_noaug
      combination = aug{level}_val{split}     e.g. aug10_val20
                  = val{split}                e.g. val20  (when no augmentation)
    """
    model_type = cfg["model_type"]
    augment    = cfg["augment"]
    aug_level  = cfg.get("aug_level", 1.0 if augment else 0.0)
    val_split  = cfg.get("val_split", cfg["train_cfg"]["val_split"])

    aug_tag = "aug" if augment else "noaug"
    group   = f"{model_type}_{aug_tag}"

    val_str     = f"val{int(round(val_split * 100)):02d}"
    aug_str     = f"aug{int(round(aug_level * 10)):02d}"
    combination = f"{aug_str}_{val_str}"

    return group, combination


def create_config_dirs(run_dir, group, combination):
    """Create run_dir/group/combination/{models,results,logs} and return config_dir."""
    config_dir = os.path.join(run_dir, group, combination)
    for s in ["models", "results", "logs"]:
        os.makedirs(os.path.join(config_dir, s), exist_ok=True)
    return config_dir


def save_hyperparams(run_dir, task_name, configs, extra=None):
    data = {
        "task": task_name,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configs": [],
    }
    for cfg in configs:
        group, combination = config_group_combination(cfg)
        entry = {
            "name":        cfg["name"],
            "group":       group,
            "combination": combination,
            "model_type":  cfg["model_type"],
            "augment":     cfg["augment"],
            "aug_level":   cfg.get("aug_level", 1.0 if cfg["augment"] else 0.0),
        }
        # Merge shared train_cfg fields (epochs, lr, batch_size, etc.)
        entry.update(cfg["train_cfg"])
        # Override val_split with the per-config value (train_cfg always has the default 0.2)
        entry["val_split"] = cfg.get("val_split", cfg["train_cfg"]["val_split"])
        if cfg.get("pretrained_ship"):
            entry["init_weights"] = "from_ship_task2"
        data["configs"].append(entry)

    if extra:
        data.update(extra)

    path = os.path.join(run_dir, "hyperparams.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Hyperparams saved: {path}")
