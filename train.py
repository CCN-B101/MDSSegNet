import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.crack_dataset import CrackDataset
from model.ML_Mutil import MLiteUNet
from model.loss import CombinedLoss
from util.utils import count_params, mIOU, precision_recall_f1_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDSSegNet independent-run training for Reviewer 4 statistical validation"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Dataset root containing train/ and val/")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset label, e.g. Crack500, DeepCrack, CTC")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, nargs=2, default=[448, 448])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/statistical")
    parser.add_argument("--result-root", type=str, default="results/statistical")
    return parser.parse_args()


def set_reproducibility(seed: int):
    """Set the random sources used by this training process."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Keep the training protocol reproducible across independent runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed DataLoader workers when num_workers > 0."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate(model, dataloader):
    """Validation metric calculation kept consistent with the original training code."""
    model.eval()
    metric = mIOU(num_classes=2)
    preds_all, gts_all = [], []

    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc="Validating", leave=False):
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            out = model(img)[0]
            pred = torch.argmax(out, dim=1)

            pred_np = pred.cpu().numpy()
            gt_np = mask.cpu().numpy()
            preds_all.append(pred_np)
            gts_all.append(gt_np)
            metric.add_batch(pred_np, gt_np)

    miou, class_ious = metric.evaluate(return_class_iou=True)

    preds_np = np.concatenate(preds_all, axis=0).reshape(-1)
    gts_np = np.concatenate(gts_all, axis=0).reshape(-1)
    precision, recall, f1, accuracy = precision_recall_f1_acc(preds_np, gts_np)

    return {
        "mIoU": float(miou),
        "ClassIoU": class_ious,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def main():
    args = parse_args()
    set_reproducibility(args.seed)

    size = tuple(args.crop_size)

    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "masks")
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")

    for p in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Required directory not found: {p}")

    # IMPORTANT:
    # - Training randomness changes with args.seed.
    # - Validation input itself must remain fixed across runs.
    #   The accompanying transform.py makes val_transform deterministic.
    trainset = CrackDataset(
        train_img_dir, train_mask_dir,
        size=size, mode="train", seed=args.seed
    )
    valset = CrackDataset(
        val_img_dir, val_mask_dir,
        size=size, mode="val", seed=None
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )

    model = MLiteUNet(
        n_classes=2,
        aux_mode="train",
        pretrained_backbone=False
    ).cuda()

    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Total Parameters: {count_params(model):.2f} M")

    model = DataParallel(model).cuda()

    criterion = CombinedLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_root) / args.dataset / f"seed_{args.seed}"
    result_dir = Path(args.result_root) / args.dataset / f"seed_{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_weight_path = checkpoint_dir / "best_mIoU.pth"
    csv_path = result_dir / "train_metrics.csv"
    log_path = result_dir / "train.log"

    headers = [
        "Epoch", "Loss", "mIoU", "Accuracy",
        "Precision", "Recall", "F1", "LearningRate"
    ]

    best_miou = -1.0
    best_epoch = -1

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile, \
         open(log_path, "w", encoding="utf-8") as logfile:

        writer = csv.writer(csvfile)
        writer.writerow(headers)

        logfile.write(f"Dataset: {args.dataset}\n")
        logfile.write(f"Seed: {args.seed}\n")
        logfile.write(f"Epochs: {args.epochs}\n")
        logfile.write(f"Batch size: {args.batch_size}\n")
        logfile.write(f"Learning rate: {args.lr}\n")
        logfile.write(f"Weight decay: {args.weight_decay}\n\n")

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            tbar = tqdm(
                enumerate(trainloader),
                total=len(trainloader),
                desc=f"{args.dataset} seed={args.seed} epoch={epoch}/{args.epochs}"
            )

            for i, (img, mask) in tbar:
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                output, out_s3, out_s4, out_s5 = model(img)

                # Correct multi-scale deep supervision:
                # main output + three genuine auxiliary outputs.
                loss1 = criterion(output, mask)
                loss2 = criterion(out_s3, mask)
                loss3 = criterion(out_s4, mask)
                loss4 = criterion(out_s5, mask)

                loss = (
                    0.5 * loss1
                    + 0.2 * loss2
                    + 0.2 * loss3
                    + 0.1 * loss4
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}")

            metrics = evaluate(model, valloader)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch,
                total_loss / len(trainloader),
                metrics["mIoU"],
                metrics["Accuracy"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1"],
                current_lr,
            ])
            csvfile.flush()

            logfile.write(
                f"Epoch {epoch:03d} | "
                f"Loss={total_loss / len(trainloader):.6f} | "
                f"mIoU={metrics['mIoU']:.6f} | "
                f"Precision={metrics['Precision']:.6f} | "
                f"Recall={metrics['Recall']:.6f} | "
                f"F1={metrics['F1']:.6f} | "
                f"Accuracy={metrics['Accuracy']:.6f} | "
                f"LR={current_lr:.8f}\n"
            )
            logfile.flush()

            # Use one fixed model-selection rule for every run:
            # best validation mIoU.
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                best_epoch = epoch
                torch.save(model.module.state_dict(), best_weight_path)

                print(
                    f"Best checkpoint updated: epoch={best_epoch}, "
                    f"val mIoU={best_miou:.6f} -> {best_weight_path}"
                )

            scheduler.step()

        logfile.write(
            f"\nTraining complete. Best epoch={best_epoch}, "
            f"Best validation mIoU={best_miou:.6f}\n"
        )

    print("\nTraining complete")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation mIoU: {best_miou:.6f}")
    print(f"Checkpoint: {best_weight_path}")
    print(f"Training log: {log_path}")
    print(f"Training CSV: {csv_path}")


if __name__ == "__main__":
    main()
