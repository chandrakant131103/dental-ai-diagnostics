"""
Convert YOLO-format labels (boxes OR polygons, normalized 0-1) into a COCO-style
JSON with per-instance `segmentation` fields, so Stage 5 (U-Net) can train
against real masks derived from your existing dental-datasetv6 labels.
"""
import argparse
import json
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def load_class_names(data_yaml_path: str):
    cfg = yaml.safe_load(Path(data_yaml_path).read_text())
    names = cfg["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


def polygon_area(xy_pairs):
    n = len(xy_pairs) // 2
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = xy_pairs[2 * i], xy_pairs[2 * i + 1]
        x2, y2 = xy_pairs[2 * ((i + 1) % n)], xy_pairs[2 * ((i + 1) % n) + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def yolo_line_to_coco_ann(values, img_w, img_h):
    if len(values) == 4:
        cx, cy, w, h = values
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        x0, y0 = cx - w / 2, cy - h / 2
        poly = [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
        bbox = [x0, y0, w, h]
        area = w * h
        return [poly], bbox, area

    if len(values) % 2 != 0 or len(values) < 6:
        return None

    xs = values[0::2]
    ys = values[1::2]
    xs_px = [x * img_w for x in xs]
    ys_px = [y * img_h for y in ys]
    poly = []
    for x, y in zip(xs_px, ys_px):
        poly.extend([x, y])

    x0, x1 = min(xs_px), max(xs_px)
    y0, y1 = min(ys_px), max(ys_px)
    bbox = [x0, y0, x1 - x0, y1 - y0]
    area = polygon_area(poly)
    return [poly], bbox, area


def main(args):
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    class_names = load_class_names(args.data_yaml)

    categories = [{"id": cid, "name": name} for cid, name in sorted(class_names.items())]

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    label_files = sorted(labels_dir.glob("*.txt"))
    skipped_no_image = 0
    skipped_bad_lines = 0

    for label_path in tqdm(label_files, desc=f"Converting {labels_dir.name}"):
        stem = label_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            skipped_no_image += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped_no_image += 1
            continue
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(round(float(parts[0])))
                values = [float(v) for v in parts[1:]]
            except ValueError:
                skipped_bad_lines += 1
                continue

            result = yolo_line_to_coco_ann(values, w, h)
            if result is None:
                skipped_bad_lines += 1
                continue
            segmentation, bbox, area = result

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco))

    print(f"[yolo_to_coco] {len(images)} images, {len(annotations)} annotations -> {out_path}")
    if skipped_no_image:
        print(f"[yolo_to_coco] WARNING: {skipped_no_image} label files had no matching image, skipped")
    if skipped_bad_lines:
        print(f"[yolo_to_coco] WARNING: {skipped_bad_lines} malformed label lines skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
