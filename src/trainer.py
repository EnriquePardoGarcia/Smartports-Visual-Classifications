import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def run_training(model, train_loader, val_loader, config, models_dir, config_name, run, logger=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_auc = -1.0
    best_model_path = None
    best_epoch = -1
    history = []
    val_loss_history = []

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_metrics["auc"] > best_val_auc
        val_loss_history.append(val_metrics["loss"])

        # Calcular std de pérdida de validación en última ventana
        val_loss_std = 0.0
        if len(val_loss_history) >= 5:
            val_loss_std = float(np.std(val_loss_history[-5:]))

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["accuracy"],
            "train_f1":   train_metrics["f1"],
            "train_auc":  train_metrics["auc"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
            "val_f1":     val_metrics["f1"],
            "val_auc":    val_metrics["auc"],
            "val_loss_std": val_loss_std,
        })

        marker = " *" if is_best else ""
        print(
            f"  Epoch {epoch:03d}/{config['epochs']} | "
            f"Train Loss {train_metrics['loss']:.4f} Acc {train_metrics['accuracy']:.4f} AUC {train_metrics['auc']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} (std={val_loss_std:.4f}) Acc {val_metrics['accuracy']:.4f} AUC {val_metrics['auc']:.4f}"
            + marker
        )

        if logger:
            logger.log_epoch(epoch, config["epochs"], train_metrics, val_metrics, is_best)

        candidate_path = os.path.join(models_dir, f"{config_name}_run{run}_epoch{epoch}.pt")
        torch.save(model.state_dict(), candidate_path)

        if is_best:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(models_dir, f"{config_name}_run{run}_best.pt")
            torch.save(model.state_dict(), best_model_path)
            os.remove(candidate_path)
        else:
            os.remove(candidate_path)

    print(f"  Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    return history, best_model_path, best_epoch
