import os
import sys
import argparse
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import PortDataset, get_transforms
from src.models import get_model
from src.evaluation import evaluate_model, full_visual_evaluation
from torch.utils.data import DataLoader

TASK_CLASSES = {
    "ship":   ["No ship", "Ship"],
    "docked": ["Undocked", "Docked"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test CSV.")
    parser.add_argument("--task",         required=True, choices=["ship", "docked"])
    parser.add_argument("--model_type",   required=True, choices=["scratch", "pretrained"])
    parser.add_argument("--model_path",   required=True, help="Path to .pt weights file.")
    parser.add_argument("--csv",          required=True, help="Path to test CSV.")
    parser.add_argument("--images_dir",   required=True, help="Path to test images folder.")
    parser.add_argument("--image_size",   type=int, default=224)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--config_name",  default=None)
    parser.add_argument("--results_dir",  default=None,
                        help="Output folder. Defaults to runs/test_<timestamp>/results/")
    parser.add_argument("--save_visuals", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    config_name = args.config_name or os.path.splitext(os.path.basename(args.model_path))[0]

    if args.results_dir:
        results_dir = args.results_dir
    else:
        from src.run_manager import create_run_dir
        base_dir = os.path.dirname(os.path.abspath(__file__))
        run_dir, _ = create_run_dir(base_dir, f"test_{args.task}")
        results_dir = os.path.join(run_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = TASK_CLASSES[args.task]

    print(f"Task        : {args.task}")
    print(f"Model type  : {args.model_type}")
    print(f"Weights     : {args.model_path}")
    print(f"Test CSV    : {args.csv}")
    print(f"Images dir  : {args.images_dir}")
    print(f"Device      : {device}")

    _, val_tf = get_transforms(augment=False, image_size=args.image_size)
    dataset = PortDataset(args.csv, args.images_dir, transform=val_tf)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

    model = get_model(args.model_type, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    result = evaluate_model(model, loader, device)

    print(f"\nResults [{config_name}]")
    print(f"  Accuracy : {result['accuracy']:.4f}")
    print(f"  F1       : {result['f1']:.4f}")
    print(f"  AUC      : {result['auc']:.4f}")

    summary = {
        "config_name": config_name,
        "task":        args.task,
        "model_type":  args.model_type,
        "model_path":  args.model_path,
        "accuracy":    result["accuracy"],
        "f1":          result["f1"],
        "auc":         result["auc"],
    }
    out_json = os.path.join(results_dir, f"{config_name}_test_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_json}")

    if args.save_visuals:
        print("\nGenerating visualizations...")
        full_visual_evaluation(
            model, loader, config_name, results_dir, class_names, device,
            result["labels"], result["preds"], result["probs"]
        )


if __name__ == "__main__":
    main()
