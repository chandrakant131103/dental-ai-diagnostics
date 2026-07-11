"""
Stage 2: Pathology segmentation training.

Usage (Colab):
    python models/segmentation/train.py \
        --images /content/data/COCO/train/images \
        --coco /content/data/COCO/annotations/train_coco.json \
        --val-images /content/data/COCO/valid/images \
        --val-coco /content/data/COCO/annotations/valid_coco.json \
        --epochs 40 --batch 8
"""
import argparse
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


def run_epoch(model, loader, optimizer, ce_loss, dice_loss, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
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
    return total_loss / len(loader.dataset)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seg-train] device={device}")

    train_ds = DentalSegDataset(args.images, args.coco, img_size=args.img_size)
    val_ds = DentalSegDataset(args.val_images, args.val_coco, img_size=args.img_size,
                               class_ids=train_ds.class_ids)
    num_classes = train_ds.num_classes
    print(f"[seg-train] {len(train_ds)} train / {len(val_ds)} val images, {num_classes} classes (incl. bg)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = UNet(n_channels=1, n_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, ce_loss, dice_loss, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, ce_loss, dice_loss, device, train=False)
        scheduler.step()
        print(f"[seg-train] epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "class_ids": train_ds.class_ids, "num_classes": num_classes},
                out_dir / "unet_best.pt",
            )
            print(f"[seg-train] saved new best -> {out_dir/'unet_best.pt'}")

    print(f"[seg-train] Done. Best val loss={best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-coco", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="runs/segment")
    args = parser.parse_args()
    main(args)
