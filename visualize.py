import os
import sys
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import PortDataset, get_transforms
from src.models import get_model
from src.evaluation import (
    evaluate_model,
    plot_predictions_grid,
    plot_errors,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_confidence_histogram,
    plot_training_history,
    plot_configs_comparison,
)

TASK_CLASSES = {
    "ship":   ["No ship", "Ship"],
    "docked": ["Undocked", "Docked"],
}

VISUALS = ["predictions", "errors", "roc", "confusion_matrix",
           "training_curves", "comparison", "confidence_hist", "all"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for any trained model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--task",         required=True, choices=["ship", "docked"])
    parser.add_argument("--visual",       required=True, choices=VISUALS,
                        help=(
                            "predictions     — grid of images with GT vs predicted label\n"
                            "errors          — only misclassified images\n"
                            "roc             — ROC curve\n"
                            "confusion_matrix— confusion matrix\n"
                            "confidence_hist — predicted probability histogram\n"
                            "training_curves — loss/acc/auc per epoch (needs --history_json)\n"
                            "comparison      — bar chart across configs (needs --results_json)\n"
                            "all             — everything available given provided args\n"
                        ))
    parser.add_argument("--model_type",   choices=["scratch", "pretrained"])
    parser.add_argument("--model_path",   help="Path to .pt weights file.")
    parser.add_argument("--csv",          help="Path to CSV.")
    parser.add_argument("--images_dir",   help="Path to images folder.")
    parser.add_argument("--history_json", help="Path to history JSON (for training_curves).")
    parser.add_argument("--results_json", help="Path to results_summary.json (for comparison).")
    parser.add_argument("--config_name",  default=None)
    parser.add_argument("--results_dir",  default=None,
                        help="Output folder. Defaults to runs/viz_<timestamp>/results/")
    parser.add_argument("--image_size",   type=int, default=224)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--n_images",     type=int, default=16)
    return parser.parse_args()


def resolve_results_dir(args):
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        return args.results_dir
    from src.run_manager import create_run_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir, _ = create_run_dir(base_dir, f"viz_{args.task}")
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


MODEL_VISUALS = {"predictions", "errors", "roc", "confusion_matrix", "confidence_hist"}


def run_model_visuals(args, visuals_to_run, results_dir):
    missing = [f"--{a}" for a, v in [
        ("model_type", args.model_type), ("model_path", args.model_path),
        ("csv", args.csv), ("images_dir", args.images_dir)
    ] if not v]
    if missing:
        print(f"ERROR: required for this visual: {', '.join(missing)}")
        sys.exit(1)

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = TASK_CLASSES[args.task]
    config_name = args.config_name or os.path.splitext(os.path.basename(args.model_path))[0]

    _, val_tf = get_transforms(augment=False, image_size=args.image_size)
    dataset   = PortDataset(args.csv, args.images_dir, transform=val_tf)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)

    model = get_model(args.model_type, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    result = evaluate_model(model, loader, device)
    print(f"  Accuracy: {result['accuracy']:.4f}  F1: {result['f1']:.4f}  AUC: {result['auc']:.4f}")

    if "predictions"      in visuals_to_run:
        plot_predictions_grid(model, loader, config_name, results_dir, class_names, device, n=args.n_images)
    if "errors"           in visuals_to_run:
        plot_errors(model, loader, config_name, results_dir, class_names, device, n=args.n_images)
    if "roc"              in visuals_to_run:
        plot_roc_curve(result["labels"], result["probs"], config_name, results_dir)
    if "confusion_matrix" in visuals_to_run:
        plot_confusion_matrix(result["labels"], result["preds"], config_name, results_dir, class_names)
    if "confidence_hist"  in visuals_to_run:
        plot_confidence_histogram(result["probs"], result["labels"], config_name, results_dir, class_names)


def main():
    args = parse_args()
    results_dir = resolve_results_dir(args)

    if args.visual == "all":
        visuals_to_run = set()
        if args.model_path and args.csv and args.images_dir and args.model_type:
            visuals_to_run |= MODEL_VISUALS
        if args.history_json:
            visuals_to_run.add("training_curves")
        if args.results_json:
            visuals_to_run.add("comparison")
        if not visuals_to_run:
            print("ERROR: for --visual all, provide at least one source of data.")
            sys.exit(1)
    else:
        visuals_to_run = {args.visual}

    if visuals_to_run & MODEL_VISUALS:
        run_model_visuals(args, visuals_to_run & MODEL_VISUALS, results_dir)

    if "training_curves" in visuals_to_run:
        if not args.history_json:
            print("ERROR: --history_json required for training_curves.")
            sys.exit(1)
        config_name = args.config_name or os.path.splitext(os.path.basename(args.history_json))[0]
        with open(args.history_json) as f:
            history = json.load(f)
        plot_training_history(history, config_name, results_dir)

    if "comparison" in visuals_to_run:
        if not args.results_json:
            print("ERROR: --results_json required for comparison.")
            sys.exit(1)
        with open(args.results_json) as f:
            all_results = json.load(f)
        plot_configs_comparison(all_results, results_dir, args.task)

    print(f"\nVisualizations saved to: {results_dir}")


if __name__ == "__main__":
    main()
