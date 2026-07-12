"""
Benchmarks end-to-end pipeline latency (detector + segmenter, excluding
disk I/O) on whatever device is available, so the README's latency row
stops being a TBD. Run this in the same environment you plan to deploy to
(Colab GPU, your server's CPU, etc) - latency is hardware-specific and a
number measured on one machine isn't a claim about another.

Usage:
    python evaluation/benchmark_latency.py \
        --detector runs/detect/tooth_localization/weights/best.pt \
        --segmenter runs/segment/unet_best.pt \
        --n-runs 50 \
        --out runs/latency_benchmark.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "models" / "segmentation"))
from unet import UNet  # noqa: E402


def benchmark(detector_path, segmenter_path, n_runs, device, imgsz=640, crop_size=256):
    detector = YOLO(detector_path)
    ckpt = torch.load(segmenter_path, map_location=device)
    segmenter = UNet(n_channels=1, n_classes=ckpt["num_classes"]).to(device)
    segmenter.load_state_dict(ckpt["model_state"])
    segmenter.eval()

    dummy_full = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    dummy_crop = torch.randn(1, 1, crop_size, crop_size).to(device)

    # warmup
    for _ in range(5):
        detector.predict(source=dummy_full, verbose=False)
        with torch.no_grad():
            segmenter(dummy_crop)

    det_times, seg_times = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        detector.predict(source=dummy_full, verbose=False)
        t1 = time.perf_counter()
        with torch.no_grad():
            segmenter(dummy_crop)
        t2 = time.perf_counter()
        det_times.append((t1 - t0) * 1000)
        seg_times.append((t2 - t1) * 1000)

    return {
        "device": device,
        "detector_ms_mean": float(np.mean(det_times)),
        "detector_ms_p95": float(np.percentile(det_times, 95)),
        "segmenter_ms_mean_per_crop": float(np.mean(seg_times)),
        "segmenter_ms_p95_per_crop": float(np.percentile(seg_times, 95)),
        "note": ("segmenter timing is per-crop; multiply by expected findings-per-image "
                 "for full-image latency. Full pipeline latency also depends on how many "
                 "teeth/findings the detector returns for a given image."),
    }


def main(args):
    result = {}
    if torch.cuda.is_available() and not args.cpu_only:
        result["gpu"] = benchmark(args.detector, args.segmenter, args.n_runs, "cuda")
    result["cpu"] = benchmark(args.detector, args.segmenter, args.n_runs, "cpu")

    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[benchmark] wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--segmenter", required=True)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--out", default="runs/latency_benchmark.json")
    args = parser.parse_args()
    main(args)
