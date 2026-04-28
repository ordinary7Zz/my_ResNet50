import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from cls_dataset import ThyroidClsDataset
from train_resnet50_cls import build_model
from utils.metrics import classification_bootstrap_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ResNet50 thyroid binary classification inference")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--label_json", type=str, required=True)
    parser.add_argument("--task_key", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=0.95)
    return parser.parse_args()


def get_device(cuda_device: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_device}")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, List[float]]:
    model.eval()

    filenames: List[str] = []
    labels: List[int] = []
    logits: List[float] = []
    probs: List[float] = []
    preds: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_labels = batch["label"].cpu().numpy().astype(np.int32)
            batch_filenames = list(batch["filename"])

            batch_logits = model(images).squeeze(1)
            batch_probs = torch.sigmoid(batch_logits)
            batch_preds = (batch_probs >= threshold).long()

            filenames.extend(batch_filenames)
            labels.extend(batch_labels.tolist())
            logits.extend(batch_logits.cpu().numpy().astype(np.float64).tolist())
            probs.extend(batch_probs.cpu().numpy().astype(np.float64).tolist())
            preds.extend(batch_preds.cpu().numpy().astype(np.int32).tolist())

    return {
        "filename": filenames,
        "label": labels,
        "logit": logits,
        "prob": probs,
        "pred": preds,
    }


def save_predictions(predictions: Dict[str, List[float]], output_path: str) -> None:
    fieldnames = ["filename", "label", "logit", "prob", "pred"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(predictions["filename"])):
            writer.writerow(
                {
                    "filename": predictions["filename"][idx],
                    "label": int(predictions["label"][idx]),
                    "logit": float(predictions["logit"][idx]),
                    "prob": float(predictions["prob"][idx]),
                    "pred": int(predictions["pred"][idx]),
                }
            )


def format_metrics(metrics_ci: Dict[str, tuple]) -> Dict[str, Dict[str, object]]:
    formatted: Dict[str, Dict[str, object]] = {}
    for name, (mean_value, ci_range) in metrics_ci.items():
        formatted[name] = {
            "mean": float(mean_value),
            "ci": [float(ci_range[0]), float(ci_range[1])],
        }
    return formatted


def save_summary(
    args: argparse.Namespace,
    dataset: ThyroidClsDataset,
    predictions: Dict[str, List[float]],
    metrics_ci: Dict[str, tuple],
    output_path: str,
) -> None:
    summary = {
        "task_key": args.task_key,
        "image_dir": args.image_dir,
        "label_json": args.label_json,
        "checkpoint_path": args.checkpoint_path,
        "num_samples": len(predictions["filename"]),
        "threshold": float(args.threshold),
        "bootstrap": {
            "n_boot": int(args.n_boot),
            "ci": float(args.ci),
        },
        "data_stats": {
            "skipped_label_minus1": int(dataset.skipped_label_minus1),
            "missing_image_files": int(dataset.missing_image_files),
        },
        "metrics": format_metrics(metrics_ci),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.cuda_device)
    dataset = ThyroidClsDataset(
        image_dir=args.image_dir,
        label_json=args.label_json,
        img_size=args.img_size,
        mode="test",
        task_key=args.task_key,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(device)
    load_checkpoint(model, args.checkpoint_path, device)

    predictions = run_inference(model, dataloader, device, args.threshold)
    metrics_ci = classification_bootstrap_metrics(
        y_probs=np.asarray(predictions["prob"], dtype=np.float32),
        y_labels=np.asarray(predictions["label"], dtype=np.int32),
        threshold=args.threshold,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=0,
    )

    save_predictions(predictions, os.path.join(args.output_dir, "predictions.csv"))
    save_summary(
        args,
        dataset,
        predictions,
        metrics_ci,
        os.path.join(args.output_dir, "metrics_summary.json"),
    )

    print(f"Saved predictions to {os.path.join(args.output_dir, 'predictions.csv')}")
    print(f"Saved summary to {os.path.join(args.output_dir, 'metrics_summary.json')}")


if __name__ == "__main__":
    main()
