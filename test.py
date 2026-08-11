import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import t as student_t
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.crack_dataset import CrackDataset
from model.ML_Mutil import MLiteUNet
from util.utils import mIOU, precision_recall_f1_acc


SEGMENTATION_METRICS = ["mIoU", "Precision", "Recall", "F1", "Accuracy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MDSSegNet 5-run test and statistical summary for Reviewer 4"
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Test root containing images/ and masks/")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset label matching the training checkpoint directory")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/statistical")
    parser.add_argument("--result-root", type=str, default="results/statistical")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--crop-size", type=int, nargs=2, default=[448, 448])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def save_prediction(pred_mask, save_path):
    pred_img = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(pred_img).save(save_path)


def evaluate_model(model, dataloader, save_dir=None):
    """
    Keep Precision/Recall/F1/Accuracy aggregation consistent with the
    original test.py: calculate per image, then average across test images.
    """
    model.eval()
    metric = mIOU(num_classes=2)

    total_prec = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_acc = 0.0
    count = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            output = model(img)[0]
            pred = torch.argmax(output, dim=1)

            pred_np = pred.cpu().numpy()
            gt_np = mask.cpu().numpy()

            metric.add_batch(pred_np, gt_np)

            p, r, f1, acc = precision_recall_f1_acc(pred.cpu(), mask.cpu())
            total_prec += p
            total_recall += r
            total_f1 += f1
            total_acc += acc
            count += 1

            if save_dir is not None:
                save_prediction(
                    pred,
                    os.path.join(save_dir, f"pred_{idx:04d}.png")
                )

    if count == 0:
        raise RuntimeError("The test loader is empty.")

    # Report percentages to match Tables 1-3.
    return {
        "mIoU": float(metric.evaluate() * 100.0),
        "Precision": float((total_prec / count) * 100.0),
        "Recall": float((total_recall / count) * 100.0),
        "F1": float((total_f1 / count) * 100.0),
        "Accuracy": float((total_acc / count) * 100.0),
    }


def summarize(values):
    """
    Sample SD (ddof=1) and two-sided 95% CI using Student's t distribution.
    For five runs, df=4 and t_0.975 is approximately 2.776.
    """
    values = np.asarray(values, dtype=float)
    n = values.size

    if n < 2:
        raise ValueError("At least two independent runs are required.")

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    t_crit = float(student_t.ppf(0.975, df=n - 1))
    margin = t_crit * sd / np.sqrt(n)

    return {
        "N": int(n),
        "Mean": mean,
        "SD": sd,
        "CI95_Lower": mean - margin,
        "CI95_Upper": mean + margin,
    }


def main():
    args = parse_args()
    size = tuple(args.crop_size)

    img_dir = os.path.join(args.data_root, "images")
    mask_dir = os.path.join(args.data_root, "masks")

    for p in [img_dir, mask_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Required test directory not found: {p}")

    # Test input must be identical for every independently trained model.
    # The accompanying transform.py makes val_transform deterministic.
    test_set = CrackDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        size=size,
        mode="val",
        seed=None
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    dataset_result_dir = Path(args.result_root) / args.dataset
    dataset_result_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    for seed in args.seeds:
        checkpoint_path = (
            Path(args.checkpoint_root)
            / args.dataset
            / f"seed_{seed}"
            / "best_mIoU.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for seed {seed}: {checkpoint_path}"
            )

        print(f"\n=== {args.dataset}: evaluating independently trained seed {seed} ===")

        model = MLiteUNet(
            n_classes=2,
            aux_mode="train",
            pretrained_backbone=False
        ).cuda()

        state_dict = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        pred_dir = None
        if args.save_predictions:
            pred_dir = dataset_result_dir / f"seed_{seed}" / "predictions"

        metrics = evaluate_model(
            model,
            test_loader,
            save_dir=str(pred_dir) if pred_dir is not None else None
        )

        run_record = {"Seed": seed, **metrics}
        all_runs.append(run_record)

        print(
            f"Seed {seed}: "
            f"mIoU={metrics['mIoU']:.4f}%, "
            f"Precision={metrics['Precision']:.4f}%, "
            f"Recall={metrics['Recall']:.4f}%, "
            f"F1={metrics['F1']:.4f}%, "
            f"Accuracy={metrics['Accuracy']:.4f}%"
        )

        del model
        torch.cuda.empty_cache()

    # Raw results from all five independent models.
    runs_csv = dataset_result_dir / "five_run_results.csv"
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Seed"] + SEGMENTATION_METRICS
        )
        writer.writeheader()
        writer.writerows(all_runs)

    # Statistical summary.
    summary_rows = []
    for metric in SEGMENTATION_METRICS:
        stats = summarize([r[metric] for r in all_runs])
        summary_rows.append({"Metric": metric, **stats})

    summary_csv = dataset_result_dir / "statistical_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Metric", "N", "Mean", "SD",
                "CI95_Lower", "CI95_Upper"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_txt = dataset_result_dir / "statistical_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(
            f"Dataset: {args.dataset}\n"
            f"Independent runs: {len(args.seeds)}\n"
            f"Seeds: {args.seeds}\n"
            "Statistics: sample SD (ddof=1), two-sided 95% CI using Student's t-distribution\n\n"
        )

        for row in summary_rows:
            f.write(
                f"{row['Metric']}: "
                f"{row['Mean']:.4f} ± {row['SD']:.4f}% "
                f"(95% CI: [{row['CI95_Lower']:.4f}, "
                f"{row['CI95_Upper']:.4f}]%)\n"
            )

    print("\n================ Statistical Summary ================")
    print(f"Dataset: {args.dataset}")
    for row in summary_rows:
        print(
            f"{row['Metric']:9s}: "
            f"{row['Mean']:.4f} ± {row['SD']:.4f}% "
            f"(95% CI [{row['CI95_Lower']:.4f}, "
            f"{row['CI95_Upper']:.4f}]%)"
        )
    print("=====================================================")
    print(f"Raw five-run results: {runs_csv}")
    print(f"Statistical CSV:      {summary_csv}")
    print(f"Statistical TXT:      {summary_txt}")


if __name__ == "__main__":
    main()
