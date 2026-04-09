import os
from datetime import datetime


class TrainingLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Training log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n")

    def _write(self, line):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_config(self, config_name, model_type, augment, train_cfg):
        self._write(f"\nCONFIGURATION: {config_name}")
        self._write("-" * 70)
        self._write(f"  model_type   : {model_type}")
        self._write(f"  augmentation : {augment}")
        self._write(f"  epochs       : {train_cfg['epochs']}")
        self._write(f"  lr           : {train_cfg['lr']}")
        self._write(f"  weight_decay : {train_cfg['weight_decay']}")
        self._write(f"  batch_size   : {train_cfg['batch_size']}")
        self._write(f"  val_split    : {train_cfg['val_split']}")
        self._write(f"  image_size   : {train_cfg['image_size']}")
        self._write(f"  n_runs       : {train_cfg['n_runs']}")
        self._write("-" * 70)

    def log_run_start(self, run, n_runs, seed):
        self._write(f"\n  Run {run}/{n_runs}  (seed={seed})")
        self._write(
            f"  {'Epoch':<8} {'TrLoss':>8} {'TrAcc':>7} {'TrF1':>7} {'TrAUC':>7} | "
            f"{'ValLoss':>8} {'ValAcc':>7} {'ValF1':>7} {'ValAUC':>7}  {'*':>3}"
        )

    def log_epoch(self, epoch, total_epochs, train_m, val_m, is_best):
        marker = "*" if is_best else ""
        self._write(
            f"  {epoch:03d}/{total_epochs:<4} "
            f"{train_m['loss']:>8.4f} {train_m['accuracy']:>7.4f} {train_m['f1']:>7.4f} {train_m['auc']:>7.4f} | "
            f"{val_m['loss']:>8.4f} {val_m['accuracy']:>7.4f} {val_m['f1']:>7.4f} {val_m['auc']:>7.4f}  {marker:>3}"
        )

    def log_run_result(self, run, acc, f1, auc, best_epoch):
        self._write(
            f"\n  Run {run} best model at epoch {best_epoch}  ->  "
            f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}"
        )

    def log_summary(self, config_name, summary):
        self._write(f"\n  SUMMARY {config_name}")
        self._write(f"  {'metric':<12} {'mean':>8} {'std':>8}")
        for k, v in summary.items():
            self._write(f"  {k:<12} {v['mean']:>8.4f} {v['std']:>8.4f}")
        self._write("=" * 70)

    def log_line(self, text):
        self._write(text)
