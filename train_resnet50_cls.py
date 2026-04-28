import os
import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, average_precision_score

from cls_dataset import ThyroidClsDataset
from utils.utils import log_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ResNet50 thyroid ultrasound binary classification")
    parser.add_argument(
        "--train_image_dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--train_label_json",
        type=str,
        required=True,
        help="JSON file with training labels (filename, malignancy, tirads)",
    )
    parser.add_argument(
        "--test_image_dirs",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories containing test images (optional)",
    )
    parser.add_argument(
        "--test_label_jsons",
        type=str,
        nargs="+",
        default=None,
        help="One or more JSON files with test labels (same order as test_image_dirs)",
    )
    parser.add_argument(
        "--test_dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional names for each test dataset (same order as test_image_dirs)",
    )

    parser.add_argument("--epoch", type=int, default=50, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay for AdamW",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="input image size (short side will be resized to this)",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index to use (default: 0)",
    )
    parser.add_argument(
        "--dir_checkpoint",
        type=str,
        default="checkpoints_resnet50",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thyroid_cls",
        help="Name used for logging/checkpoint folder",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    return parser.parse_args()


def build_model(device: torch.device) -> nn.Module:
    # Use ImageNet-pretrained ResNet50 and adapt the final layer for binary classification.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # binary (benign vs malignant)
    model.to(device)
    return model


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, float, float, float, float, float]:
    """Compute accuracy, precision, recall, F1, AUROC, AUPRC (all in [0,1])."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        labels_long = labels.long()

        tp = ((preds == 1) & (labels_long == 1)).sum().item()
        tn = ((preds == 0) & (labels_long == 0)).sum().item()
        fp = ((preds == 1) & (labels_long == 0)).sum().item()
        fn = ((preds == 0) & (labels_long == 1)).sum().item()

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # Convert to numpy for sklearn metrics
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        labels_np = labels_long.detach().cpu().numpy().astype(np.int64)

        # AUROC and AUPRC are only defined when both classes are present
        if len(np.unique(labels_np)) == 2:
            try:
                auroc = float(roc_auc_score(labels_np, probs_np))
            except Exception:
                auroc = 0.0
            try:
                auprc = float(average_precision_score(labels_np, probs_np))
            except Exception:
                auprc = 0.0
        else:
            auroc = 0.0
            auprc = 0.0

    return acc, precision, recall, f1, auroc, auprc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, precision, recall, f1, auroc, auprc = compute_metrics(logits_cat, labels_cat)
    avg_loss = total_loss / total_samples
    return avg_loss, acc, precision, recall, f1, auroc, auprc


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("logs_resnet50", args.dataset_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f"resnet50_{args.dataset_name}_log.log")
    log_file = open(log_path, "w", encoding="utf-8")

    try:
        log_print(f"Training started at {time.ctime()}\n", log_file)
        log_print(f"Device: {device}\n", log_file)
        log_print(f"Train image dir: {args.train_image_dir}\n", log_file)
        log_print(f"Train label json: {args.train_label_json}\n", log_file)
        log_print(f"Test image dirs: {args.test_image_dirs}\n", log_file)
        log_print(f"Test label jsons: {args.test_label_jsons}\n", log_file)
        log_print(f"Epochs: {args.epoch}, Batch size: {args.batch_size}, LR: {args.lr}\n", log_file)

        # Dataset & dataloader (train only; no validation set)
        train_dataset = ThyroidClsDataset(
            image_dir=args.train_image_dir,
            label_json=args.train_label_json,
            img_size=args.img_size,
            mode="train",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
        )

        # Optional multiple test datasets
        test_loaders = []
        if args.test_image_dirs is not None and args.test_label_jsons is not None:
            if len(args.test_image_dirs) != len(args.test_label_jsons):
                raise ValueError(
                    "test_image_dirs and test_label_jsons must have the same length."
                )
            if args.test_dataset_names is not None and len(args.test_dataset_names) != len(
                args.test_image_dirs
            ):
                raise ValueError(
                    "test_dataset_names (if provided) must have the same length as test_image_dirs."
                )

            for idx, (img_dir, lbl_json) in enumerate(
                zip(args.test_image_dirs, args.test_label_jsons)
            ):
                name = (
                    args.test_dataset_names[idx]
                    if args.test_dataset_names is not None
                    else os.path.basename(os.path.normpath(img_dir)) or f"test_{idx}"
                )
                ds = ThyroidClsDataset(
                    image_dir=img_dir,
                    label_json=lbl_json,
                    img_size=args.img_size,
                    mode="test",
                )
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=8,
                    drop_last=False,
                )
                test_loaders.append((name, dl))

        model = build_model(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = opt.AdamW(
            [{"params": model.parameters(), "initial_lr": args.lr}],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)

        # checkpoint directory
        ckpt_dir = os.path.join(args.dir_checkpoint, args.dataset_name, timestamp)
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(args.epoch):
            model.train()
            running_loss = 0.0
            running_samples = 0

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                bs = labels.size(0)
                running_loss += loss.item() * bs
                running_samples += bs

            scheduler.step()

            train_loss = running_loss / max(running_samples, 1)
            msg = f"Epoch {epoch + 1}/{args.epoch} - train_loss: {train_loss:.4f}"

            log_print(msg + "\n", log_file)

            # periodic checkpoints
            if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epoch - 1:
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"resnet50_{args.dataset_name}_epoch_{epoch + 1}.pth",
                )
                torch.save(model.state_dict(), ckpt_path)
                log_print(f"Checkpoint saved at {ckpt_path}\n", log_file)

        # ===== Test evaluation after training (final epoch model, multiple test sets) =====
        for name, test_loader in test_loaders:
            test_loss, acc, precision, recall, f1, auroc, auprc = evaluate(
                model, test_loader, device, criterion
            )
            log_print(
                f"Test results [{name}] - "
                f"loss: {test_loss:.4f}, "
                f"acc: {acc:.4f}, prec: {precision:.4f}, "
                f"rec: {recall:.4f}, f1: {f1:.4f}, "
                f"auroc: {auroc:.4f}, auprc: {auprc:.4f}\n",
                log_file,
            )

        log_print(f"Training completed at {time.ctime()}\n", log_file)

    finally:
        log_file.close()


if __name__ == "__main__":
    main()

