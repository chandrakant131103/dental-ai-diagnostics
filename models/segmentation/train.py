"""
Stage 2: Pathology segmentation training.

FIXES applied (see dataset.py docstring for the root-cause writeup):
  1. dataset.py now yields per-finding crops that match inference, instead
     of whole panoramic images squashed to img_size - this alone should
     fix most of the "dead logits / 0% positive Grad-CAM" collapse.
  2. CrossEntropyLoss is now class-weighted (inverse pixel frequency),
     combined with the existing Dice loss, so background (still the
     majority class even within a padded crop) doesn't dominate the
     gradient the way it likely did before.
  3. A mandatory sanity check runs before training starts: it prints the
     unique mask values and positive-pixel fraction for a sample of
     batches, and refuses to start training if masks are empty (all
     background) - so a data/label bug fails loudly instead of silently
     producing a collapsed model 40 epochs later.
  4. Per-class Dice is logged every epoch (not just aggregate loss), so
     you can see e.g. "Caries" specifically failing instead of only a
     single scalar going down.

Usage (Colab):
    python models/segmentation/train.py \
        --images /content/data/COCO/train/images \
        --coco /content/data/COCO/annotations/train_coco.json \
        --val-images /content/data/COCO/valid/images \
        --val-coco /content/data/COCO/annotations/valid_coco.json \
        --epochs 40 --batch 8
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DentalSegDataset
from unet import UNet


class DiceLoss(nn.Module):
    """Combined with cross-entropy: CE handles per-pixel classification,
    Dice directly optimizes overlap and is far more robust to the heavy
    background-vs-lesion imbalance every pixel-wise medical seg task has."""

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = nn.functional.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def dice_per_class(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)
    scores = []
    for c in range(1, num_classes):  # skip background
        pred_c = (preds == c)
        tgt_c = (targets == c)
        inter = (pred_c & tgt_c).sum().item()
        union = pred_c.sum().item() + tgt_c.sum().item()
        scores.append(1.0 if union == 0 else 2 * inter / union)
    return scores


def sanity_check(loader, num_classes, n_batches: int = 5):
    """Fail loudly, before spending an hour training, if masks are empty
    or malformed. This is the check recommended after the U-Net was found
    predicting all-background: verify positive-class pixels actually reach
    the loss function before assuming it's purely a loss-weighting issue.
    """
    print("[seg-train] Running pre-training sanity check on mask contents...")
    total_pixels = 0
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.int64)
    seen = 0
    for imgs, masks in loader:
        vals = torch.unique(masks)
        print(f"  batch {seen}: mask unique values = {vals.tolist()}, "
              f"positive-pixel fraction = {(masks > 0).float().mean().item():.4%}")
        for v in vals.tolist():
            if 0 <= v < num_classes:
                class_pixel_counts[v] += (masks == v).sum().item()
        total_pixels += masks.numel()
        seen += 1
        if seen >= n_batches:
            break

    positive_frac = 1.0 - (class_pixel_counts[0].item() / max(total_pixels, 1))
    print(f"[seg-train] Sanity check: {positive_frac:.4%} of sampled pixels are non-background.")
    if positive_frac == 0.0:
        raise RuntimeError(
            "[seg-train] ABORTING: 0% of pixels in the first "
            f"{n_batches} batches belong to any lesion class. This means either "
            "(a) the COCO segmentation field is empty/degenerate for these "
            "images - check data/yolo_to_coco.py output counts, or "
            "(b) class_ids / category id remapping is wrong. Fix this before "
            "training - it is the exact failure mode that produced dead "
            "logits and 0% positive Grad-CAM last time."
        )
    if positive_frac < 0.005:
        print("[seg-train] WARNING: positive-pixel fraction is very low (<0.5%). "
              "Class weighting will help, but also consider tighter crop_padding "
              "in DentalSegDataset so lesions occupy more of each crop.")


def run_epoch(model, loader, optimizer, ce_loss, dice_loss, device, num_classes, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    dice_sums = None
    n_batches = 0
    with torch.set_grad_enabled(train):
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)

            batch_dice = dice_per_class(logits.detach(), masks, num_classes)
            dice_sums = batch_dice if dice_sums is None else [a + b for a, b in zip(dice_sums, batch_dice)]
            n_batches += 1

    mean_dice = [d / max(n_batches, 1) for d in dice_sums] if dice_sums else []
    return total_loss / len(loader.dataset), mean_dice


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seg-train] device={device}")

    train_ds = DentalSegDataset(args.images, args.coco, img_size=args.img_size,
                                 crop_padding=args.crop_padding)
    val_ds = DentalSegDataset(args.val_images, args.val_coco, img_size=args.img_size,
                               class_ids=train_ds.class_ids, crop_padding=args.crop_padding)
    num_classes = train_ds.num_classes
    print(f"[seg-train] {len(train_ds)} train / {len(val_ds)} val crops, {num_classes} classes (incl. bg)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    sanity_check(train_loader, num_classes)

    if args.no_class_weights:
        class_weights = None
        print("[seg-train] Class weighting disabled via --no-class-weights.")
    else:
        print("[seg-train] Computing class pixel weights (inverse frequency)...")
        class_weights = train_ds.compute_class_pixel_weights().to(device)
        class_names = ["background"] + [str(c) for c in train_ds.class_ids]
        for name, w in zip(class_names, class_weights.tolist()):
            print(f"  {name:<20} weight={w:.3f}")

    model = UNet(n_channels=1, n_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, ce_loss, dice_loss,
                                            device, num_classes, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, optimizer, ce_loss, dice_loss,
                                        device, num_classes, train=False)
        scheduler.step()

        mean_val_dice = sum(val_dice) / len(val_dice) if val_dice else 0.0
        print(f"[seg-train] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_mean_dice={mean_val_dice:.4f}")
        print(f"  per-class val dice ({[str(c) for c in train_ds.class_ids]}): "
              f"{[round(d, 3) for d in val_dice]}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            "val_mean_dice": mean_val_dice, "val_dice_per_class": val_dice,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "class_ids": train_ds.class_ids, "num_classes": num_classes},
                out_dir / "unet_best.pt",
            )
            print(f"[seg-train] saved new best -> {out_dir/'unet_best.pt'}")

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"[seg-train] Done. Best val loss={best_val:.4f}. History saved to {out_dir/'training_history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-coco", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--crop-padding", type=float, default=0.25,
                         help="fraction of box size padded on each side when cropping a finding")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--out", default="runs/segment")
    args = parser.parse_args()
    main(args)
