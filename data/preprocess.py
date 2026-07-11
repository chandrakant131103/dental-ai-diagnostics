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
  4. (If --labels-src given) remaps YOLO-format bounding box labels to match
     the letterboxed image, in the same worker pass as the image itself.

Parallelized across CPU cores with multiprocessing, and resumable: if you
rerun after a Colab disconnect, already-processed images are skipped rather
than redone, so you don't lose progress.

Usage:
    python data/preprocess.py --src /content/data/train/images \
                               --dst /content/data_proc/train/images \
                               --labels-src /content/data/train/labels \
                               --labels-dst /content/data_proc/train/labels \
                               --size 1024
"""
import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")


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


def denoise(img: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """bilateral (default) is ~5-10x faster than NLM and still edge-preserving
    - the right tradeoff for Colab free-tier CPU limits. Use method='nlm' for
    marginally higher quality if you have more CPU headroom / more patience."""
    if method == "nlm":
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 6, 7, 21)
    # bilateral: d=5 (small neighborhood) keeps it fast; sigmaColor/sigmaSpace
    # tuned for grayscale-ish X-ray noise without over-smoothing lesion edges
    return cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)


def letterbox(img: np.ndarray, size: int = 1024, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), color, dtype=np.uint8) if img.ndim == 3 else np.full(
        (size, size), color[0], dtype=np.uint8
    )
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top: top + nh, left: left + nw] = resized
    return canvas, scale, left, top


def adjust_yolo_labels_text(label_text: str, orig_w: int, orig_h: int, scale: float,
                             pad_left: float, pad_top: float, size: int) -> str:
    """Handles two label line formats found in the wild:
      - box:     class cx cy w h                (5 tokens)
      - polygon: class x1 y1 x2 y2 ... xn yn     (odd token count > 5, YOLO-seg style)

    This dataset's "YOLO" folder ships polygon segmentation labels, not plain
    boxes - discovered when every remapped label file came out empty because
    the original code assumed a fixed 5-value box format and silently errored
    on every polygon line. Since Stage 1 is a plain object detector (Stage 2's
    U-Net already handles precise lesion shape via the separate COCO masks),
    polygons are converted here to their tight bounding box.
    """
    lines_out = []
    for line in label_text.splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        cls = int(tokens[0])
        coords = [float(t) for t in tokens[1:]]

        if len(coords) == 4:
            # already a box: cx, cy, w, h
            cx, cy, w, h = coords
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # polygon: pairs of (x, y) -> tight bounding box
            xs = coords[0::2]
            ys = coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            print(f"[preprocess] WARNING: unrecognized label line format, skipping: '{line[:60]}...'")
            continue

        # de-normalize corners against original image size
        x1_px, y1_px = x1 * orig_w, y1 * orig_h
        x2_px, y2_px = x2 * orig_w, y2 * orig_h

        # apply the same scale + pad as the image (letterbox)
        x1_px = x1_px * scale + pad_left
        y1_px = y1_px * scale + pad_top
        x2_px = x2_px * scale + pad_left
        y2_px = y2_px * scale + pad_top

        # re-normalize against the letterboxed canvas, back to YOLO box format
        cx_new = ((x1_px + x2_px) / 2) / size
        cy_new = ((y1_px + y2_px) / 2) / size
        w_new = (x2_px - x1_px) / size
        h_new = (y2_px - y1_px) / size

        lines_out.append(f"{cls} {cx_new:.6f} {cy_new:.6f} {w_new:.6f} {h_new:.6f}")
    return "\n".join(lines_out) + ("\n" if lines_out else "")


def _worker(args):
    img_path, dst_img_path, label_path, out_label_path, size, denoise_method = args
    try:
        # resume support: skip if already done
        if dst_img_path.exists() and (out_label_path is None or out_label_path.exists()):
            return ("skipped", str(img_path))

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return ("error", f"could not read {img_path}")
        orig_h, orig_w = img.shape[:2]

        img = denoise(img, method=denoise_method)
        img = apply_clahe(img)
        img, scale, pad_left, pad_top = letterbox(img, size=size)

        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_img_path), img)

        if out_label_path is not None:
            out_label_path.parent.mkdir(parents=True, exist_ok=True)
            label_text = label_path.read_text() if (label_path and label_path.exists()) else ""
            remapped = adjust_yolo_labels_text(label_text, orig_w, orig_h, scale, pad_left, pad_top, size)
            out_label_path.write_text(remapped)

        return ("ok", str(img_path))
    except Exception as e:
        return ("error", f"{img_path}: {e}")


def process_dir(src_dir, dst_dir, labels_src=None, labels_dst=None, size=1024,
                 workers=None, progress_every=200, denoise_method="bilateral"):
    src = Path(src_dir)
    dst = Path(dst_dir)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS])
    total = len(files)
    print(f"[preprocess] {total} images in {src} -> {dst} (size={size}, denoise={denoise_method})")

    tasks = []
    for p in files:
        dst_img_path = dst / p.name
        label_path = Path(labels_src) / (p.stem + ".txt") if labels_src else None
        out_label_path = Path(labels_dst) / (p.stem + ".txt") if labels_dst else None
        tasks.append((p, dst_img_path, label_path, out_label_path, size, denoise_method))

    workers = workers or max(1, os.cpu_count())
    print(f"[preprocess] Using {workers} worker processes")

    done, skipped, errors = 0, 0, 0
    start = time.time()

    with mp.Pool(processes=workers) as pool:
        for status, info in pool.imap_unordered(_worker, tasks, chunksize=8):
            done += 1
            if status == "skipped":
                skipped += 1
            elif status == "error":
                errors += 1
                print(f"[preprocess] ERROR: {info}")

            if done % progress_every == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                print(
                    f"[preprocess] processed {done}/{total} "
                    f"(skipped={skipped}, errors={errors}) "
                    f"- {rate:.1f} img/s - ~{remaining/60:.1f} min remaining",
                    flush=True,
                )

    print(f"[preprocess] Done -> {dst} ({done} processed, {skipped} skipped, {errors} errors, "
          f"{(time.time()-start)/60:.1f} min total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="images dir")
    parser.add_argument("--dst", required=True, help="output images dir")
    parser.add_argument("--labels-src", default=None, help="optional YOLO labels dir to remap")
    parser.add_argument("--labels-dst", default=None, help="output labels dir (required if --labels-src set)")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=None, help="default: cpu_count - 1")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--denoise", choices=["bilateral", "nlm"], default="bilateral",
                         help="bilateral (default, fast) or nlm (slower, marginally cleaner)")
    args = parser.parse_args()

    if args.labels_src and not args.labels_dst:
        parser.error("--labels-dst is required when --labels-src is given")

    process_dir(
        args.src, args.dst,
        labels_src=args.labels_src, labels_dst=args.labels_dst,
        size=args.size, workers=args.workers, progress_every=args.progress_every,
        denoise_method=args.denoise,
    )
