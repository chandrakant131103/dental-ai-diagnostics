"""
Preprocessing for panoramic dental X-rays.

Panoramic X-rays are low-contrast, often noisy, and vary a lot in exposure.
This module applies:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization) - standard in
     medical imaging to bring out subtle findings (caries, bone loss) without
     blowing out already-bright regions.
  2. Denoising (fast non-local means) - panoramic X-rays are grainy.
  3. Resize + pad to a consistent square size (letterbox) so YOLO/U-Net see
     undistorted teeth geometry.

Usage:
    python data/preprocess.py --src /content/data/train/images \
                               --dst /content/data_processed/train/images
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def apply_clahe(img: np.ndarray, clip_limit: float = 2.5, tile_grid_size=(8, 8)) -> np.ndarray:
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)


def denoise(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, 6, 7, 21)


def letterbox(img: np.ndarray, size: int = 1024, color=(114, 114, 114)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), color, dtype=np.uint8) if img.ndim == 3 else np.full(
        (size, size), color[0], dtype=np.uint8
    )
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def process_image(path: Path, out_path: Path, size: int = 1024):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[preprocess] Could not read {path}, skipping.")
        return
    img = denoise(img)
    img = apply_clahe(img)
    img = letterbox(img, size=size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def process_dir(src_dir: str, dst_dir: str, size: int = 1024, exts=(".jpg", ".jpeg", ".png")):
    src = Path(src_dir)
    dst = Path(dst_dir)
    files = [p for p in src.iterdir() if p.suffix.lower() in exts]
    print(f"[preprocess] Found {len(files)} images in {src}")
    for p in tqdm(files):
        process_image(p, dst / p.name, size=size)
    print(f"[preprocess] Done -> {dst}")


def adjust_yolo_labels(label_path: Path, out_path: Path, orig_w: int, orig_h: int, size: int = 1024):
    """Letterboxing changes where objects sit in the frame, so YOLO-format
    labels (normalized cx,cy,w,h) must be remapped to match. Run this in
    lockstep with process_image on the corresponding label file."""
    scale = size / max(orig_w, orig_h)
    nw, nh = orig_w * scale, orig_h * scale
    pad_left = (size - nw) / 2
    pad_top = (size - nh) / 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not label_path.exists():
        out_path.write_text("")
        return

    lines_out = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()
        cls, cx, cy, w, h = int(cls), float(cx), float(cy), float(w), float(h)

        cx_px, cy_px = cx * orig_w, cy * orig_h
        w_px, h_px = w * orig_w, h * orig_h

        cx_px = cx_px * scale + pad_left
        cy_px = cy_px * scale + pad_top
        w_px *= scale
        h_px *= scale

        lines_out.append(
            f"{cls} {cx_px / size:.6f} {cy_px / size:.6f} {w_px / size:.6f} {h_px / size:.6f}"
        )
    out_path.write_text("\n".join(lines_out) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="images dir")
    parser.add_argument("--dst", required=True, help="output images dir")
    parser.add_argument("--labels-src", default=None, help="optional YOLO labels dir to remap")
    parser.add_argument("--labels-dst", default=None, help="output labels dir")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    process_dir(args.src, args.dst, size=args.size)

    if args.labels_src:
        src_img_dir = Path(args.src)
        for img_path in src_img_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_path = Path(args.labels_src) / (img_path.stem + ".txt")
            out_label_path = Path(args.labels_dst) / (img_path.stem + ".txt")
            adjust_yolo_labels(label_path, out_label_path, w, h, size=args.size)
        print(f"[preprocess] Remapped labels -> {args.labels_dst}")
