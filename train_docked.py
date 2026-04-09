import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import build_dataloaders
from src.models import get_model
from src.trainer import run_training
from src.logger import TrainingLogger
from src.evaluation import (
    evaluate_model, full_visual_evaluation,
    plot_training_history, plot_configs_comparison, save_history_json
)
from src.run_manager import create_run_dir, create_config_dirs, save_hyperparams, config_group_combination

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MATERIAL_DIR  = os.path.join(BASE_DIR, "P1-Material")
IMAGES_DIR    = os.path.join(MATERIAL_DIR, "images")
CSV_PATH      = os.path.join(MATERIAL_DIR, "docked.csv")
SHIP_RUNS_DIR = os.path.join(BASE_DIR, "runs")

FINETUNE_CONFIG = {
    "epochs": 25, "lr": 1e-4, "weight_decay": 1e-4,
    "batch_size": 32, "val_split": 0.2, "image_size": 224, "n_runs": 3,
}

# Pretrained sin augmentation (3 configs: 3 val_splits)
# Pretrained con augmentation (6 configs: 2 aug_level x 3 val_split)
CONFIGS = [
    # --- Pretrained sin augmentation (3 configs) ---
    {"name": "docked_noaug_val10",  "model_type": "pretrained", "augment": False, "aug_level": 0.0, "val_split": 0.1, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_noaug_val20",  "model_type": "pretrained", "augment": False, "aug_level": 0.0, "val_split": 0.2, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_noaug_val30",  "model_type": "pretrained", "augment": False, "aug_level": 0.0, "val_split": 0.3, "train_cfg": FINETUNE_CONFIG},

    # --- Pretrained con augmentation (6 configs: 2 aug_level x 3 val_split) ---
    {"name": "docked_aug10_val10",  "model_type": "pretrained", "augment": True,  "aug_level": 1.0, "val_split": 0.1, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_aug10_val20",  "model_type": "pretrained", "augment": True,  "aug_level": 1.0, "val_split": 0.2, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_aug10_val30",  "model_type": "pretrained", "augment": True,  "aug_level": 1.0, "val_split": 0.3, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_aug15_val10",  "model_type": "pretrained", "augment": True,  "aug_level": 1.5, "val_split": 0.1, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_aug15_val20",  "model_type": "pretrained", "augment": True,  "aug_level": 1.5, "val_split": 0.2, "train_cfg": FINETUNE_CONFIG},
    {"name": "docked_aug15_val30",  "model_type": "pretrained", "augment": True,  "aug_level": 1.5, "val_split": 0.3, "train_cfg": FINETUNE_CONFIG},
]


def find_best_ship_model():
    if not os.path.isdir(SHIP_RUNS_DIR):
        return None
    ship_runs = sorted([
        d for d in os.listdir(SHIP_RUNS_DIR)
        if d.startswith("ship_") and os.path.isdir(os.path.join(SHIP_RUNS_DIR, d))
    ])
    for run_folder in reversed(ship_runs):
        run_path = os.path.join(SHIP_RUNS_DIR, run_folder)
        for group_candidate in ["pretrained_aug", "pretrained_noaug"]:
            group_path = os.path.join(run_path, group_candidate)
            if not os.path.isdir(group_path):
                continue
            for combo_folder in sorted(os.listdir(group_path)):
                models_dir = os.path.join(group_path, combo_folder, "models")
                if not os.path.isdir(models_dir):
                    continue
                candidates = [f for f in os.listdir(models_dir) if f.endswith("_best.pt")]
                if candidates:
                    return os.path.join(models_dir, sorted(candidates)[0])
    return None


def load_ship_weights(model, ship_path, device):
    state_dict = torch.load(ship_path, map_location=device)
    model_dict = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    print(f"  Loaded {len(compatible)}/{len(model_dict)} layers from: {ship_path}")
    return model


def run_config(cfg, run_dir, ship_path):
    print(f"\n{'='*60}\nConfig: {cfg['name']}\n{'='*60}")

    group, combination = config_group_combination(cfg)
    config_dir  = create_config_dirs(run_dir, group, combination)
    models_dir  = os.path.join(config_dir, "models")
    results_dir = os.path.join(config_dir, "results")
    logs_dir    = os.path.join(config_dir, "logs")
    tcfg        = cfg["train_cfg"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Undocked", "Docked"]

    logger = TrainingLogger(os.path.join(logs_dir, "training.txt"))
    logger.log_config(cfg["name"], cfg["model_type"], cfg["augment"], tcfg)
    logger.log_line(f"  init_weights : {ship_path}")

    all_run_metrics = {"accuracy": [], "f1": [], "auc": []}

    for run in range(1, tcfg["n_runs"] + 1):
        seed = 41 + run
        print(f"\n--- Run {run}/{tcfg['n_runs']} (seed={seed}) ---")
        logger.log_run_start(run, tcfg["n_runs"], seed)

        train_loader, val_loader = build_dataloaders(
            CSV_PATH, IMAGES_DIR,
            augment=cfg["augment"],
            val_split=cfg.get("val_split", tcfg["val_split"]),
            batch_size=tcfg["batch_size"],
            image_size=tcfg["image_size"],
            seed=seed,
            aug_level=cfg.get("aug_level", 1.0),
        )

        model = get_model(cfg["model_type"], num_classes=1)
        model = load_ship_weights(model, ship_path, device)

        history, best_path, best_epoch = run_training(
            model, train_loader, val_loader, tcfg,
            models_dir, cfg["name"], run, logger=logger
        )

        save_history_json(history, os.path.join(results_dir, f"run{run}_history.json"))

        if run == 1:
            plot_training_history(history, cfg["name"], results_dir)

        model.load_state_dict(torch.load(best_path, map_location=device))
        result = evaluate_model(model, val_loader, device)

        if run == 1:
            full_visual_evaluation(
                model, val_loader, cfg["name"], results_dir,
                class_names, device,
                result["labels"], result["preds"], result["probs"]
            )

        for k in all_run_metrics:
            all_run_metrics[k].append(result[k])

        logger.log_run_result(run, result["accuracy"], result["f1"], result["auc"], best_epoch)
        print(f"  Run {run}: Acc={result['accuracy']:.4f} F1={result['f1']:.4f} AUC={result['auc']:.4f}")

    summary = {}
    print(f"\nSummary [{cfg['name']}]:")
    for k, vals in all_run_metrics.items():
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {k}: mean={summary[k]['mean']:.4f}  std={summary[k]['std']:.4f}")
        if summary[k]['std'] > 0.05:
            print(f"    ⚠️  HIGH STD detected - consider more data augmentation")

    logger.log_summary(cfg["name"], summary)

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_final_report(all_results, configs, run_dir):
    cfg_by_name = {c["name"]: c for c in configs}
    all_names   = list(all_results.keys())
    metrics     = ["accuracy", "f1", "auc"]

    def mean(name, m):    return all_results[name][m]["mean"]
    def std(name, m):     return all_results[name][m]["std"]
    def robustness(name): return np.mean([std(name, m) for m in metrics])

    groups = {
        "pretrained_noaug": [n for n in all_names if not cfg_by_name[n]["augment"]],
        "pretrained_aug":   [n for n in all_names if     cfg_by_name[n]["augment"]],
    }
    group_titles = {
        "pretrained_noaug": "Pretrained sin augmentation",
        "pretrained_aug":   "Pretrained con augmentation",
    }

    lines = []
    L = lines.append

    L("=" * 70)
    L("INFORME FINAL DE EXPERIMENTOS - DOCKED DETECTION (Fine-tuning)")
    L("=" * 70)
    L("")
    L("Este informe recoge los resultados de todos los entrenamientos y analiza")
    L("que configuracion ha funcionado mejor en cada aspecto.")
    L("")

    # ------------------------------------------------------------------ #
    # 1. TABLA GLOBAL DE RESULTADOS
    # ------------------------------------------------------------------ #
    L("=" * 70)
    L("1. RESULTADOS GLOBALES (mean +/- std sobre 3 runs)")
    L("=" * 70)
    L(f"{'Config':<28} {'Accuracy':>14} {'F1':>14} {'AUC':>14} {'Robustez*':>10}")
    L("-" * 80)
    for name in all_names:
        rob = robustness(name)
        L(f"{name:<28} "
          f"{mean(name,'accuracy'):>6.4f}+/-{std(name,'accuracy'):.4f}  "
          f"{mean(name,'f1'):>6.4f}+/-{std(name,'f1'):.4f}  "
          f"{mean(name,'auc'):>6.4f}+/-{std(name,'auc'):.4f}  "
          f"{rob:>9.4f}")
    L("")
    L("  * Robustez = std medio de las 3 metricas (menor = mas estable)")
    L("")

    # ------------------------------------------------------------------ #
    # 2. ANALISIS POR GRUPO
    # ------------------------------------------------------------------ #
    L("=" * 70)
    L("2. ANALISIS POR GRUPO")
    L("=" * 70)

    for gkey, gnames in groups.items():
        L("")
        L(f"--- {group_titles[gkey]} ({len(gnames)} configs) ---")
        L("")

        L(f"  {'Config':<28} {'Acc mean':>9} {'F1 mean':>9} {'AUC mean':>9} {'Robustez':>9}")
        L("  " + "-" * 60)
        for name in gnames:
            L(f"  {name:<28} {mean(name,'accuracy'):>9.4f} {mean(name,'f1'):>9.4f} "
              f"{mean(name,'auc'):>9.4f} {robustness(name):>9.4f}")
        L("")

        best_acc  = max(gnames, key=lambda n: mean(n, "accuracy"))
        best_f1   = max(gnames, key=lambda n: mean(n, "f1"))
        best_auc  = max(gnames, key=lambda n: mean(n, "auc"))
        best_rob  = min(gnames, key=robustness)
        worst_rob = max(gnames, key=robustness)

        L(f"  Mejor accuracy : {best_acc}  ({mean(best_acc,'accuracy'):.4f} +/- {std(best_acc,'accuracy'):.4f})")
        L(f"  Mejor F1       : {best_f1}  ({mean(best_f1,'f1'):.4f} +/- {std(best_f1,'f1'):.4f})")
        L(f"  Mejor AUC      : {best_auc}  ({mean(best_auc,'auc'):.4f} +/- {std(best_auc,'auc'):.4f})")
        L(f"  Mas robusto    : {best_rob}  (std medio = {robustness(best_rob):.4f})")
        L(f"  Mas sensible   : {worst_rob}  (std medio = {robustness(worst_rob):.4f})")

        val_splits = sorted(set(cfg_by_name[n]["val_split"] for n in gnames))
        if len(val_splits) > 1:
            L("")
            L("  Comparacion por val_split (media de todas las configs del grupo con ese split):")
            for vs in val_splits:
                subset = [n for n in gnames if cfg_by_name[n]["val_split"] == vs]
                avg_acc = np.mean([mean(n, "accuracy") for n in subset])
                avg_rob = np.mean([robustness(n) for n in subset])
                L(f"    val_split={vs:.1f}  ->  acc_media={avg_acc:.4f}  robustez_media={avg_rob:.4f}")
            best_vs_acc = max(val_splits, key=lambda vs: np.mean([mean(n, "accuracy") for n in gnames if cfg_by_name[n]["val_split"] == vs]))
            best_vs_rob = min(val_splits, key=lambda vs: np.mean([robustness(n) for n in gnames if cfg_by_name[n]["val_split"] == vs]))
            L(f"    => Mejor val_split por accuracy  : {best_vs_acc}")
            L(f"    => Mejor val_split por robustez  : {best_vs_rob}")

        aug_levels = sorted(set(cfg_by_name[n]["aug_level"] for n in gnames if cfg_by_name[n]["augment"]))
        if len(aug_levels) > 1:
            L("")
            L("  Comparacion por aug_level (media de todas las configs del grupo con ese nivel):")
            for al in aug_levels:
                subset = [n for n in gnames if cfg_by_name[n]["aug_level"] == al]
                avg_acc = np.mean([mean(n, "accuracy") for n in subset])
                avg_rob = np.mean([robustness(n) for n in subset])
                L(f"    aug_level={al:.1f}  ->  acc_media={avg_acc:.4f}  robustez_media={avg_rob:.4f}")
            best_al_acc = max(aug_levels, key=lambda al: np.mean([mean(n, "accuracy") for n in gnames if cfg_by_name[n]["aug_level"] == al]))
            best_al_rob = min(aug_levels, key=lambda al: np.mean([robustness(n) for n in gnames if cfg_by_name[n]["aug_level"] == al]))
            L(f"    => Mejor aug_level por accuracy  : {best_al_acc}")
            L(f"    => Mejor aug_level por robustez  : {best_al_rob}")

    # ------------------------------------------------------------------ #
    # 3. COMPARACION ENTRE GRUPOS
    # ------------------------------------------------------------------ #
    L("")
    L("=" * 70)
    L("3. COMPARACION ENTRE LOS 2 GRUPOS")
    L("=" * 70)
    L("")

    group_avg = {}
    for gkey, gnames in groups.items():
        group_avg[gkey] = {
            "accuracy": np.mean([mean(n, "accuracy") for n in gnames]),
            "f1":       np.mean([mean(n, "f1")       for n in gnames]),
            "auc":      np.mean([mean(n, "auc")       for n in gnames]),
            "robustez": np.mean([robustness(n)        for n in gnames]),
        }

    L(f"  {'Grupo':<30} {'Acc media':>10} {'F1 media':>10} {'AUC media':>10} {'Robustez':>10}")
    L("  " + "-" * 65)
    for gkey in groups:
        g = group_avg[gkey]
        L(f"  {group_titles[gkey]:<30} {g['accuracy']:>10.4f} {g['f1']:>10.4f} {g['auc']:>10.4f} {g['robustez']:>10.4f}")
    L("")

    best_group_acc = max(groups, key=lambda g: group_avg[g]["accuracy"])
    best_group_rob = min(groups, key=lambda g: group_avg[g]["robustez"])

    aug_acc   = group_avg["pretrained_aug"]["accuracy"]
    noaug_acc = group_avg["pretrained_noaug"]["accuracy"]
    aug_rob   = group_avg["pretrained_aug"]["robustez"]
    noaug_rob = group_avg["pretrained_noaug"]["robustez"]

    L(f"  Con aug vs Sin aug (accuracy media):  con_aug={aug_acc:.4f}  sin_aug={noaug_acc:.4f}")
    L(f"    => La augmentation {'MEJORA' if aug_acc >= noaug_acc else 'NO mejora'} la accuracy media")
    L(f"  Con aug vs Sin aug (robustez):        con_aug={aug_rob:.4f}  sin_aug={noaug_rob:.4f}")
    L(f"    => La augmentation {'MEJORA' if aug_rob <= noaug_rob else 'NO mejora'} la robustez (std {'menor' if aug_rob <= noaug_rob else 'mayor'})")
    L("")
    L(f"  Grupo con mejor accuracy media  : {group_titles[best_group_acc]}  ({group_avg[best_group_acc]['accuracy']:.4f})")
    L(f"  Grupo mas robusto               : {group_titles[best_group_rob]}  (std medio = {group_avg[best_group_rob]['robustez']:.4f})")

    # ------------------------------------------------------------------ #
    # 4. GANADORES ABSOLUTOS
    # ------------------------------------------------------------------ #
    L("")
    L("=" * 70)
    L("4. GANADORES ABSOLUTOS (sobre todas las configs)")
    L("=" * 70)
    L("")

    winner_acc = max(all_names, key=lambda n: mean(n, "accuracy"))
    winner_f1  = max(all_names, key=lambda n: mean(n, "f1"))
    winner_auc = max(all_names, key=lambda n: mean(n, "auc"))
    winner_rob = min(all_names, key=robustness)
    loser_rob  = max(all_names, key=robustness)

    all_vs = sorted(set(cfg_by_name[n]["val_split"] for n in all_names))
    vs_acc = {vs: np.mean([mean(n, "accuracy") for n in all_names if cfg_by_name[n]["val_split"] == vs]) for vs in all_vs}
    vs_rob = {vs: np.mean([robustness(n)        for n in all_names if cfg_by_name[n]["val_split"] == vs]) for vs in all_vs}
    best_vs_global_acc = max(all_vs, key=lambda vs: vs_acc[vs])
    best_vs_global_rob = min(all_vs, key=lambda vs: vs_rob[vs])

    aug_names = [n for n in all_names if cfg_by_name[n]["augment"]]
    all_al = sorted(set(cfg_by_name[n]["aug_level"] for n in aug_names))
    al_acc = {al: np.mean([mean(n, "accuracy") for n in aug_names if cfg_by_name[n]["aug_level"] == al]) for al in all_al}
    al_rob = {al: np.mean([robustness(n)        for n in aug_names if cfg_by_name[n]["aug_level"] == al]) for al in all_al}
    best_al_global_acc = max(all_al, key=lambda al: al_acc[al])
    best_al_global_rob = min(all_al, key=lambda al: al_rob[al])

    L(f"  Mejor accuracy  : {winner_acc}")
    L(f"    Acc={mean(winner_acc,'accuracy'):.4f}+/-{std(winner_acc,'accuracy'):.4f}  "
      f"F1={mean(winner_acc,'f1'):.4f}  AUC={mean(winner_acc,'auc'):.4f}")
    L("")
    L(f"  Mejor F1        : {winner_f1}")
    L(f"    F1={mean(winner_f1,'f1'):.4f}+/-{std(winner_f1,'f1'):.4f}  "
      f"Acc={mean(winner_f1,'accuracy'):.4f}  AUC={mean(winner_f1,'auc'):.4f}")
    L("")
    L(f"  Mejor AUC       : {winner_auc}")
    L(f"    AUC={mean(winner_auc,'auc'):.4f}+/-{std(winner_auc,'auc'):.4f}  "
      f"Acc={mean(winner_auc,'accuracy'):.4f}  F1={mean(winner_auc,'f1'):.4f}")
    L("")
    L(f"  Mas robusto     : {winner_rob}  (std medio={robustness(winner_rob):.4f})")
    L(f"    Acc={mean(winner_rob,'accuracy'):.4f}+/-{std(winner_rob,'accuracy'):.4f}  "
      f"F1={mean(winner_rob,'f1'):.4f}+/-{std(winner_rob,'f1'):.4f}  "
      f"AUC={mean(winner_rob,'auc'):.4f}+/-{std(winner_rob,'auc'):.4f}")
    L("")
    L(f"  Mas sensible    : {loser_rob}  (std medio={robustness(loser_rob):.4f})")
    L(f"    Acc={mean(loser_rob,'accuracy'):.4f}+/-{std(loser_rob,'accuracy'):.4f}  "
      f"F1={mean(loser_rob,'f1'):.4f}+/-{std(loser_rob,'f1'):.4f}  "
      f"AUC={mean(loser_rob,'auc'):.4f}+/-{std(loser_rob,'auc'):.4f}")
    L("")
    L(f"  Mejor val_split por accuracy : {best_vs_global_acc}  "
      f"(acc_media={vs_acc[best_vs_global_acc]:.4f} sobre {len([n for n in all_names if cfg_by_name[n]['val_split']==best_vs_global_acc])} configs)")
    L(f"  Mejor val_split por robustez : {best_vs_global_rob}  "
      f"(std_medio={vs_rob[best_vs_global_rob]:.4f})")
    L("")
    if len(all_al) > 1:
        L(f"  Mejor aug_level por accuracy : {best_al_global_acc}  "
          f"(acc_media={al_acc[best_al_global_acc]:.4f} sobre configs con aug)")
        L(f"  Mejor aug_level por robustez : {best_al_global_rob}  "
          f"(std_medio={al_rob[best_al_global_rob]:.4f})")
        L("")

    # ------------------------------------------------------------------ #
    # 5. CONCLUSION FINAL
    # ------------------------------------------------------------------ #
    L("=" * 70)
    L("5. CONCLUSION FINAL")
    L("=" * 70)
    L("")
    best_overall = winner_acc
    best_cfg_meta = cfg_by_name[best_overall]
    L(f"  La configuracion con mejor rendimiento global es: {best_overall}")
    L(f"  Tipo de modelo  : {best_cfg_meta['model_type']}")
    L(f"  Augmentation    : {'Si (aug_level=' + str(best_cfg_meta['aug_level']) + ')' if best_cfg_meta['augment'] else 'No'}")
    L(f"  Val split       : {best_cfg_meta['val_split']}")
    L(f"  Accuracy media  : {mean(best_overall,'accuracy'):.4f} +/- {std(best_overall,'accuracy'):.4f}")
    L(f"  F1 media        : {mean(best_overall,'f1'):.4f} +/- {std(best_overall,'f1'):.4f}")
    L(f"  AUC media       : {mean(best_overall,'auc'):.4f} +/- {std(best_overall,'auc'):.4f}")
    L("")
    L(f"  El modelo mas robusto (mas estable entre runs) es: {winner_rob}")
    if winner_rob != best_overall:
        rob_meta = cfg_by_name[winner_rob]
        L(f"  Tipo: {rob_meta['model_type']}  |  Aug: {'Si level=' + str(rob_meta['aug_level']) if rob_meta['augment'] else 'No'}  |  Val: {rob_meta['val_split']}")
        L(f"  Tiene peor accuracy ({mean(winner_rob,'accuracy'):.4f}) pero es el mas consistente entre ejecuciones.")
    else:
        L(f"  Coincide con la mejor configuracion por accuracy.")
    L("")

    verdict_aug = "ayuda" if aug_acc >= noaug_acc else "no ayuda en este dataset"
    verdict_vs  = str(best_vs_global_acc)
    L("  Valoracion general:")
    L(f"    - La data augmentation {verdict_aug} a mejorar la accuracy en fine-tuning.")
    if aug_rob <= noaug_rob:
        L(f"    - La augmentation REDUCE la variabilidad entre runs (mejora robustez).")
    else:
        L(f"    - La augmentation NO reduce la variabilidad entre runs en este caso.")
    L(f"    - El mejor val_split global por accuracy es {verdict_vs}.")
    if best_vs_global_rob != best_vs_global_acc:
        L(f"    - Pero el mas robusto es val_split={best_vs_global_rob}, hay un trade-off.")
    if len(all_al) > 1:
        if best_al_global_acc == best_al_global_rob:
            L(f"    - El aug_level {best_al_global_acc} es mejor tanto en accuracy como en robustez.")
        else:
            L(f"    - aug_level {best_al_global_acc} gana en accuracy pero aug_level {best_al_global_rob} es mas robusto.")
    L("")
    L("=" * 70)

    report_path = os.path.join(run_dir, "informe_final.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nInforme final guardado en: {report_path}")
    return report_path


def main():
    ship_path = find_best_ship_model()
    if not ship_path:
        print("ERROR: no ship model found in runs/. Run train_ship.py first.")
        sys.exit(1)
    print(f"Using ship weights: {ship_path}")

    run_dir, run_name = create_run_dir(BASE_DIR, "docked")
    print(f"Run directory: {run_dir}")

    config_info = {
        "description": "Fine-tuning experiments with different validation splits and augmentation levels",
        "metrics_tracked": ["validation_loss_stability", "std_across_runs", "robustness"],
        "guidance": "Higher std indicates model is sensitive - increase augmentation. High val_loss std indicates sudden changes.",
    }

    save_hyperparams(run_dir, "docked", CONFIGS, extra={
        "csv": CSV_PATH,
        "images_dir": IMAGES_DIR,
        "ship_weights": ship_path,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "experiment_info": config_info,
    })

    all_results = {}
    for cfg in CONFIGS:
        all_results[cfg["name"]] = run_config(cfg, run_dir, ship_path)

    global_results_path = os.path.join(run_dir, "results_summary.json")
    with open(global_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    global_results_dir = os.path.join(run_dir, "results")
    os.makedirs(global_results_dir, exist_ok=True)
    plot_configs_comparison(all_results, global_results_dir, "docked")
    generate_final_report(all_results, CONFIGS, run_dir)

    print(f"\nRun complete. All outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()