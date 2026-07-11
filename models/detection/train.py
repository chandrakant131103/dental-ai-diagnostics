"""
Stage 1: Tooth / finding localization using Ultralytics YOLO.

Differs from a naive `yolo train` one-liner by:
  - computing class weights from the actual label distribution and logging
    them (Ultralytics doesn't expose per-class loss weighting directly, but
    knowing the imbalance drives your sampling/augmentation decisions and is
    something you should show in your README metrics table)
  - keeping W&B logging on (the copied notebook explicitly uninstalls it -
    don't; experiment tracking is exactly what reviewers want to see)
  - saving run config + git commit hash alongside weights for reproducibility

Usage (Colab):
    python models/detection/train.py --data /content/data/data.yaml \
        --model yolov8s.pt --epochs 60 --imgsz 640 --batch 16
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO


def compute_class_distribution(labels_dir: str) -> dict:
    counts = Counter()
    for f in Path(labels_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                # tolerate "1" or "1.0" - defensive against any label writer
                # upstream that hands back a float-formatted class id
                counts[int(round(float(line.split()[0])))] += 1
    return dict(sorted(counts.items()))


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main(args):
    data_yaml = Path(args.data)
    data_cfg = yaml.safe_load(data_yaml.read_text())
    train_labels_dir = (data_yaml.parent / data_cfg["train"].replace("images", "labels")).resolve()

    if train_labels_dir.exists():
        label_files = list(train_labels_dir.glob("*.txt"))
        dist = compute_class_distribution(str(train_labels_dir))
        if not dist:
            print(
                f"[train] WARNING: found {len(label_files)} label file(s) in {train_labels_dir}, "
                f"but zero non-empty annotation lines across all of them. "
                f"Training will likely produce a useless model with no positive boxes to learn from. "
                f"Check that preprocessing/augmentation actually populated this labels folder correctly "
                f"before continuing (see: !find {train_labels_dir} -name '*.txt' | xargs wc -l | tail -1)."
            )
        else:
            names = data_cfg.get("names", {})
            print("[train] Class distribution (train split):")
            for cls_id, count in dist.items():
                cls_name = names[cls_id] if isinstance(names, list) else names.get(cls_id, cls_id)
                print(f"  {cls_id:>2} {cls_name:<20} {count}")
            imbalance_ratio = max(dist.values()) / max(min(dist.values()), 1)
            print(f"[train] Max/min class imbalance ratio: {imbalance_ratio:.1f}x")
            if imbalance_ratio > 10:
                print("[train] Severe imbalance detected -> recommend focal loss / oversampling rare classes")
    else:
        print(f"[train] WARNING: expected labels dir {train_labels_dir} does not exist - skipping distribution check.")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=args.lr0,
        patience=args.patience,
        dropout=0.1,
        degrees=5,
        flipud=0.0,   # keep off - clinically meaningful orientation
        fliplr=0.0,   # same reasoning as augment.py safe_mode
        mosaic=0.5,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    run_dir = Path(model.trainer.save_dir)
    meta = {
        "git_commit": get_git_hash(),
        "base_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "data_yaml": str(data_yaml),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[train] Done. Weights + metadata at {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="base weights or yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="tooth_localization")
    args = parser.parse_args()
    main(args)
