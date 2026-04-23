import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)


def evaluate_model(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc,
            "labels": all_labels, "preds": all_preds, "probs": all_probs}


def _denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def plot_training_history(history, config_name, results_dir):
    epochs = [h["epoch"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    # Detectar cambios bruscos en validación
    val_diffs = np.abs(np.diff(val_losses))
    max_diff = np.max(val_diffs) if len(val_diffs) > 0 else 0
    avg_diff = np.mean(val_diffs) if len(val_diffs) > 0 else 0

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfica 1: Loss (train vs val)
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="train", linewidth=2)
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], label="val", linewidth=2)
    axes[0, 0].set_title(f"{config_name} — Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Gráfica 2: Accuracy
    axes[0, 1].plot(epochs, [h["train_acc"] for h in history], label="train", linewidth=2)
    axes[0, 1].plot(epochs, [h["val_acc"] for h in history], label="val", linewidth=2)
    axes[0, 1].set_title(f"{config_name} — Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gráfica 3: AUC
    axes[1, 0].plot(epochs, [h["train_auc"] for h in history], label="train", linewidth=2)
    axes[1, 0].plot(epochs, [h["val_auc"] for h in history], label="val", linewidth=2)
    axes[1, 0].set_title(f"{config_name} — AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Gráfica 4: Std de validación y cambios bruscos
    if len(history) > 0 and "val_loss_std" in history[0]:
        val_stds = [h.get("val_loss_std", 0.0) for h in history]
        axes[1, 1].plot(epochs, val_stds, marker="o", label="Loss std (5-epoch window)", linewidth=2)
        axes[1, 1].axhline(y=avg_diff, color="r", linestyle="--", label=f"Avg change: {avg_diff:.4f}")
        axes[1, 1].axhline(y=max_diff, color="orange", linestyle="--", label=f"Max change: {max_diff:.4f}")
        axes[1, 1].set_title(f"{config_name} — Validation Loss Stability (std & changes)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{config_name} — Training Curves", fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_training.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves saved: {path}")
    print(f"    Validation stability: max_change={max_diff:.4f}, avg_change={avg_diff:.4f}")


def plot_confusion_matrix(labels, preds, config_name, results_dir, class_names=None):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(config_name)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_cm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {path}")


def plot_predictions_grid(model, loader, config_name, results_dir, class_names, device, n=16):
    model.eval()
    images_list, labels_list, preds_list, probs_list = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            for i in range(len(images)):
                images_list.append(images[i])
                labels_list.append(int(labels[i]))
                preds_list.append(int(preds[i]))
                probs_list.append(float(probs[i]))
            if len(images_list) >= n * 2:
                break

    n = min(n, len(images_list))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for idx in range(n):
        img   = _denormalize(images_list[idx]).permute(1, 2, 0).numpy()
        gt    = labels_list[idx]
        pred  = preds_list[idx]
        prob  = probs_list[idx]
        color = "green" if gt == pred else "red"
        ax = axes[idx]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"GT: {class_names[gt]}\nPred: {class_names[pred]} ({prob:.2f})",
                     fontsize=8, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{config_name} — predictions (green=correct, red=wrong)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Predictions grid saved: {path}")


def plot_errors(model, loader, config_name, results_dir, class_names, device, n=16):
    model.eval()
    errors = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            for i in range(len(images)):
                gt   = int(labels[i])
                pred = int(preds[i])
                if gt != pred:
                    errors.append((images[i], gt, pred, float(probs[i])))

    if not errors:
        print(f"  No errors found for {config_name}.")
        return

    n = min(n, len(errors))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for idx in range(n):
        img_tensor, gt, pred, prob = errors[idx]
        img = _denormalize(img_tensor).permute(1, 2, 0).numpy()
        ax = axes[idx]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"GT: {class_names[gt]}\nPred: {class_names[pred]} ({prob:.2f})",
                     fontsize=8, color="red")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"{config_name} — errors ({len(errors)} total)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_errors.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Errors grid saved: {path}")


def plot_roc_curve(labels, probs, config_name, results_dir):
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
    except ValueError:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{config_name} — ROC curve")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_roc.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {path}")


def plot_confidence_histogram(probs, labels, config_name, results_dir, class_names):
    probs  = np.array(probs)
    labels = np.array(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.hist(probs[mask], bins=20, alpha=0.6, label=f"GT: {cls_name}", range=(0, 1))
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("Predicted probability (class 1)")
    ax.set_ylabel("Count")
    ax.set_title(f"{config_name} — confidence distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_confidence_hist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confidence histogram saved: {path}")


def plot_configs_comparison(all_results, results_dir, task_name):
    configs = list(all_results.keys())
    x = np.arange(len(configs))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["accuracy", "f1", "auc"]):
        means = [all_results[c][metric]["mean"] for c in configs]
        stds  = [all_results[c][metric]["std"]  for c in configs]
        bars  = ax.bar(x, means, width=0.6, yerr=stds, capsize=4,
                       color=colors[:len(configs)])
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.4)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.01, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"{task_name} — comparison (mean ± std)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{task_name}_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Comparison plot saved: {path}")


def plot_correct_and_errors(model, loader, config_name, results_dir, class_names, device):
    """Generates a single figure with two sections: correct and wrong predictions."""
    model.eval()
    correct, errors = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs   = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds   = (probs >= 0.5).astype(int)
            for i in range(len(images)):
                gt, pred, prob = int(labels[i]), int(preds[i]), float(probs[i])
                entry = (_denormalize(images[i]).permute(1, 2, 0).numpy(), gt, pred, prob)
                if gt == pred:
                    correct.append(entry)
                else:
                    errors.append(entry)

    n_correct = len(correct)
    n_errors  = len(errors)
    n_total   = n_correct + n_errors

    if n_total == 0:
        return

    cols = max(max(n_correct, n_errors, 1), 4)
    rows = 2  # row 0 = correct, row 1 = errors
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    if rows == 1:
        axes = [axes]

    # Section headers via row titles
    section_labels = [
        f"CORRECTAS ({n_correct})",
        f"FALLIDAS ({n_errors})",
    ]
    section_colors = ["green", "red"]
    sections       = [correct, errors]

    for row, (section, title, color) in enumerate(zip(sections, section_labels, section_colors)):
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(title, fontsize=11, fontweight="bold",
                              color=color, rotation=90, labelpad=10)
            if col < len(section):
                img, gt, pred, prob = section[col]
                ax.imshow(img)
                ax.set_title(
                    f"Real: {class_names[gt]}\nPred: {class_names[pred]} ({prob:.2f})",
                    fontsize=8, color=color
                )
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)

    fig.suptitle(f"{config_name} — resultados ({n_correct}/{n_total} correctas)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{config_name}_correct_vs_errors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Correct vs errors saved: {path}")


def full_visual_evaluation(model, loader, config_name, results_dir, class_names, device,
                            labels, preds, probs):
    plot_correct_and_errors(model, loader, config_name, results_dir, class_names, device)
    plot_errors(model, loader, config_name, results_dir, class_names, device, n=16)
    plot_roc_curve(labels, probs, config_name, results_dir)
    plot_confusion_matrix(labels, preds, config_name, results_dir, class_names)
    plot_confidence_histogram(probs, labels, config_name, results_dir, class_names)


def save_history_json(history, path):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
