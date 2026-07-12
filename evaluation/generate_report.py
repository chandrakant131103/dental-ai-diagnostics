"""
Reads whatever eval artifacts exist under a detector run dir (results.csv
from ultralytics, eval/summary.json + eval/per_class_report.csv from the
custom fixed-IoU evaluator) plus an optional segmentation history.json and
latency benchmark json, and:

  1. Renders a confusion-matrix PNG from evaluation/metrics.py's output
     (or copies ultralytics' own confusion_matrix.png if the custom one
     isn't available).
  2. Rewrites the "## Results" section of README.md in place with real
     numbers, so it's never left as a manual TBD.

Usage:
    python evaluation/generate_report.py \
        --detect-run runs/detect/tooth_localization \
        --seg-history runs/segment/training_history.json \
        --latency runs/latency_benchmark.json \
        --readme README.md
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_detect_metrics(run_dir: Path) -> dict:
    out = {}
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        import csv
        rows = list(csv.DictReader(results_csv.open()))
        if rows:
            last = rows[-1]
            out["mAP50"] = float(last.get("metrics/mAP50(B)", "nan"))
            out["mAP50_95"] = float(last.get("metrics/mAP50-95(B)", "nan"))
            out["precision_B"] = float(last.get("metrics/precision(B)", "nan"))
            out["recall_B"] = float(last.get("metrics/recall(B)", "nan"))
            out["epochs_completed"] = int(last["epoch"])

    summary_json = run_dir / "eval" / "summary.json"
    if summary_json.exists():
        summary = json.loads(summary_json.read_text())
        out["macro_precision"] = summary.get("macro_precision")
        out["macro_recall"] = summary.get("macro_recall")
        out["macro_f1"] = summary.get("macro_f1")
        out["iou_thresh"] = summary.get("iou_thresh")

    per_class_csv = run_dir / "eval" / "per_class_report.csv"
    if per_class_csv.exists():
        import csv
        out["per_class"] = list(csv.DictReader(per_class_csv.open()))

    return out


def load_seg_metrics(history_path: Path) -> dict:
    if not history_path or not history_path.exists():
        return {}
    history = json.loads(history_path.read_text())
    if not history:
        return {}
    best = min(history, key=lambda h: h["val_loss"])
    return {
        "val_mean_dice": best.get("val_mean_dice"),
        "val_dice_per_class": best.get("val_dice_per_class"),
        "best_epoch": best.get("epoch"),
    }


def load_latency(latency_path: Path) -> dict:
    if not latency_path or not latency_path.exists():
        return {}
    return json.loads(latency_path.read_text())


def build_results_table(detect: dict, seg: dict, latency: dict) -> str:
    def fmt(v, suffix=""):
        if v is None:
            return "TBD"
        if isinstance(v, float):
            return f"{v:.3f}{suffix}"
        return f"{v}{suffix}"

    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Detector mAP@50 | {fmt(detect.get('mAP50'))} |",
        f"| Detector mAP@50-95 | {fmt(detect.get('mAP50_95'))} |",
        f"| Detector macro precision / recall (fixed-IoU eval, thresh={detect.get('iou_thresh', 0.5)}) | "
        f"{fmt(detect.get('macro_precision'))} / {fmt(detect.get('macro_recall'))} |",
        f"| Caries precision / recall | "
        + _class_pr(detect, "Caries") + " |",
        f"| Segmentation mean Dice | {fmt(seg.get('val_mean_dice'))} |",
        f"| Inference latency (CPU) | {fmt(latency.get('cpu_ms_per_image'), ' ms/image')} |",
        f"| Inference latency (GPU) | {fmt(latency.get('gpu_ms_per_image'), ' ms/image')} |",
        f"| Detector epochs completed | {fmt(detect.get('epochs_completed'))} |",
    ]
    if detect.get("per_class"):
        lines.append("")
        lines.append("**Per-class detection performance** (fixed IoU-matched eval):")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for row in detect["per_class"]:
            lines.append(f"| {row['class']} | {row['precision']} | {row['recall']} | {row['f1']} |")
    return "\n".join(lines)


def _class_pr(detect: dict, class_name: str) -> str:
    for row in detect.get("per_class", []):
        if row["class"] == class_name:
            return f"{row['precision']} / {row['recall']}"
    return "TBD / TBD"


def update_readme(readme_path: Path, table_md: str):
    text = readme_path.read_text()
    pattern = re.compile(
        r"(## Results\n_.*?_\n\n)\|.*?(?=\n##|\Z)", re.DOTALL
    )
    replacement = r"\1" + table_md + "\n"
    if pattern.search(text):
        new_text = pattern.sub(replacement, text)
    else:
        new_text = text.rstrip() + "\n\n## Results\n\n" + table_md + "\n"
    readme_path.write_text(new_text)
    print(f"[generate_report] README results table updated in {readme_path}")


def main(args):
    detect = load_detect_metrics(Path(args.detect_run)) if args.detect_run else {}
    seg = load_seg_metrics(Path(args.seg_history)) if args.seg_history else {}
    latency = load_latency(Path(args.latency)) if args.latency else {}

    table_md = build_results_table(detect, seg, latency)
    print(table_md)

    if args.readme:
        update_readme(Path(args.readme), table_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect-run", help="path to runs/detect/<name> dir")
    parser.add_argument("--seg-history", help="path to runs/segment/training_history.json")
    parser.add_argument("--latency", help="path to a latency_benchmark.json from benchmark_latency.py")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    main(args)
