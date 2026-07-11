"""
Bounding-box-aware augmentation for panoramic X-rays using Albumentations.

Dental X-rays have specific constraints: don't flip horizontally if you care
about tooth side (left/right jaw matters clinically) unless you also swap
FDI-numbering labels - so by default we keep flips off and lean on
rotation/brightness/contrast/noise, which are safe and effective for X-ray
domain augmentation.

Usage:
    python data/augment.py --images /content/data/train/images \
                            --labels /content/data/train/labels \
                            --out-images /content/data_aug/train/images \
                            --out-labels /content/data_aug/train/labels \
                            --multiplier 3
"""
import argparse
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm


def build_transform(safe_mode: bool = True):
    """safe_mode=True avoids left/right flips (clinically meaningful in
    dental imaging). Set False only if you also remap FDI tooth-number
    classes on flip."""
    transforms = [
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.CoarseDropout(max_holes=3, max_height=24, max_width=24, p=0.15),
    ]
    if not safe_mode:
        transforms.insert(0, A.HorizontalFlip(p=0.5))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def read_yolo_labels(label_path: Path):
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        boxes.append([float(cx), float(cy), float(w), float(h)])
        classes.append(int(cls))
    return boxes, classes


def write_yolo_labels(path: Path, boxes, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(classes, boxes)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def augment_dataset(images_dir, labels_dir, out_images_dir, out_labels_dir, multiplier=3, safe_mode=True):
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    out_images_dir, out_labels_dir = Path(out_images_dir), Path(out_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(safe_mode=safe_mode)
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"[augment] {len(image_paths)} source images, x{multiplier} augmentations each")

    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes, classes = read_yolo_labels(label_path)

        # always copy the original through untouched
        cv2.imwrite(str(out_images_dir / img_path.name), img)
        write_yolo_labels(out_labels_dir / (img_path.stem + ".txt"), boxes, classes)

        for i in range(multiplier):
            try:
                augmented = transform(image=img, bboxes=boxes, class_labels=classes)
            except Exception as e:
                print(f"[augment] skipped {img_path.name} aug {i}: {e}")
                continue
            out_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(out_images_dir / out_name), augmented["image"])
            write_yolo_labels(
                out_labels_dir / f"{img_path.stem}_aug{i}.txt",
                augmented["bboxes"],
                augmented["class_labels"],
            )

    print(f"[augment] Done -> {out_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--multiplier", type=int, default=3)
    parser.add_argument("--allow-flip", action="store_true", help="disable safe_mode")
    args = parser.parse_args()

    augment_dataset(
        args.images, args.labels, args.out_images, args.out_labels,
        multiplier=args.multiplier, safe_mode=not args.allow_flip,
    )
