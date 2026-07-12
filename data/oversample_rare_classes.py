"""
Oversamples training images containing rare classes (e.g. Caries) by
duplicating them (as symlinks, so no extra disk space) N times in the
train split, before running models/detection/train.py.

WHY: Ultralytics YOLOv8's public train() API doesn't expose per-class loss
weighting (no cls_pw-style knob like YOLOv5 had). The practical, supported
way to fix a class like Caries being drowned out by Filling/Root Canal is
to change the *sampling frequency* - show the model images containing
Caries more often per epoch - rather than the loss function.

This targets the actual number from the eval report: Caries recall was
0.16 (1005 false negatives vs 189 true positives) while Filling/Root Canal
each have 5-10x more training instances. Oversampling caries-containing
images by ~4-6x roughly rebalances the images-per-class ratio without
touching the loss.

Usage:
    python data/oversample_rare_classes.py \
        --images /content/data_aug/train/images \
        --labels /content/data_aug/train/labels \
        --data-yaml /content/data_aug/data.yaml \
        --target-classes Caries \
        --factor 5 \
        --out-images /content/data_oversampled/train/images \
        --out-labels /content/data_oversampled/train/labels

Then point models/detection/train.py --data at a data.yaml whose `train:`
field is the new out-images directory (copy data.yaml and edit that one
field, or pass --write-data-yaml to have this script do it for you).
"""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def load_class_names(data_yaml_path: str) -> dict:
    cfg = yaml.safe_load(Path(data_yaml_path).read_text())
    names = cfg["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def find_images_with_classes(labels_dir: Path, target_ids: set) -> set:
    hits = set()
    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            cls_id = int(round(float(line.split()[0])))
            if cls_id in target_ids:
                hits.add(label_file.stem)
                break
    return hits


def main(args):
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    name_to_id = {v: k for k, v in load_class_names(args.data_yaml).items()}
    target_ids = set()
    for name in args.target_classes:
        if name not in name_to_id:
            raise ValueError(f"Class '{name}' not found in data.yaml names: {list(name_to_id)}")
        target_ids.add(name_to_id[name])

    rare_stems = find_images_with_classes(labels_dir, target_ids)
    print(f"[oversample] {len(rare_stems)} images contain target classes {args.target_classes}")

    all_images = list(images_dir.iterdir())
    copied, dup_count = 0, 0
    for img_path in all_images:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        # always include the original once
        _link_or_copy(img_path, out_images / img_path.name, args.copy)
        _link_or_copy(label_path, out_labels / label_path.name, args.copy)
        copied += 1

        if stem in rare_stems:
            for i in range(1, args.factor):
                dup_img = out_images / f"{stem}__dup{i}{img_path.suffix}"
                dup_lbl = out_labels / f"{stem}__dup{i}.txt"
                _link_or_copy(img_path, dup_img, args.copy)
                _link_or_copy(label_path, dup_lbl, args.copy)
                dup_count += 1

    print(f"[oversample] Wrote {copied} base images + {dup_count} duplicate copies "
          f"of rare-class images to {out_images}")

    if args.write_data_yaml:
        cfg = yaml.safe_load(Path(args.data_yaml).read_text())
        cfg["train"] = str(out_images.resolve())
        new_yaml_path = Path(args.data_yaml).parent / "data_oversampled.yaml"
        new_yaml_path.write_text(yaml.dump(cfg))
        print(f"[oversample] Wrote new data yaml -> {new_yaml_path} "
              f"(pass this to models/detection/train.py --data)")


def _link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # symlinks aren't always available (e.g. some Windows setups) - fall back
            shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--target-classes", nargs="+", default=["Caries"])
    parser.add_argument("--factor", type=int, default=5,
                         help="how many total copies of each rare-class image (1 = no oversampling)")
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--copy", action="store_true",
                         help="copy files instead of symlinking (use on filesystems without symlink support)")
    parser.add_argument("--write-data-yaml", action="store_true")
    args = parser.parse_args()
    main(args)
