"""
Evaluates a trained U-Net checkpoint on a val set: per-class and mean Dice,
written to runs/segment/eval_summary.json. This script didn't exist before -
train.py logged per-epoch Dice during training but there was no standalone
way to re-evaluate a saved checkpoint against a held-out set.

Usage:
    python models/segmentation/evaluate.py \
        --checkpoint runs/segment/unet_best.pt \
        --val-images /content/data/COCO/valid/images \
        --val-coco /content/data/COCO/annotations/valid_coco.json \
        --out runs/segment/eval_summary.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DentalSegDataset
from unet import UNet


def dice_per_class(preds, targets, num_classes):
    scores = []
    for c in range(1, num_classes):
        pred_c = (preds == c)
        tgt_c = (targets == c)
        inter = (pred_c & tgt_c).sum().item()
        union = pred_c.sum().item() + tgt_c.sum().item()
        scores.append(None if union == 0 else 2 * inter / union)
    return scores


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_ids = ckpt["class_ids"]
    num_classes = ckpt["num_classes"]

    model = UNet(n_channels=1, n_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = DentalSegDataset(args.val_images, args.val_coco, class_ids=class_ids)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    per_class_sums = [[] for _ in range(num_classes - 1)]
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            scores = dice_per_class(preds, masks, num_classes)
            for i, s in enumerate(scores):
                if s is not None:
                    per_class_sums[i].append(s)

    per_class_mean = [sum(v) / len(v) if v else None for v in per_class_sums]
    valid_scores = [v for v in per_class_mean if v is not None]
    mean_dice = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    result = {
        "mean_dice": mean_dice,
        "per_class_dice": dict(zip([str(c) for c in class_ids], per_class_mean)),
        "n_val_crops": len(val_ds),
    }
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[seg-eval] wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-coco", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default="runs/segment/eval_summary.json")
    args = parser.parse_args()
    main(args)
