"""
Downloads the Dental X-Ray Panoramic dataset from Kaggle into /content/data
(Colab) or ./data/raw (local).

Usage (Colab):
    from google.colab import files
    files.upload()   # upload kaggle.json here first
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !python data/kaggle_download.py --dataset reemsalahshehab/dental-datasetv6 --out /content/data
"""
import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def download_dataset(dataset_slug: str, out_dir: str, unzip: bool = True) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(out_path)]
    if unzip:
        cmd.append("--unzip")

    print(f"[kaggle_download] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed.\nstdout: {result.stdout}\nstderr: {result.stderr}\n"
            "Make sure ~/.kaggle/kaggle.json exists with valid API credentials."
        )
    print(result.stdout)
    return out_path


def verify_yolo_structure(root: Path):
    """Sanity check that train/valid/test + images/labels + data.yaml exist."""
    expected_splits = ["train", "valid", "test"]
    missing = []
    for split in expected_splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            missing.append(str(root / split))
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        missing.append(str(yaml_path))

    if missing:
        print(f"[kaggle_download] WARNING - missing expected paths: {missing}")
    else:
        n_train = len(list((root / "train" / "images").glob("*")))
        n_valid = len(list((root / "valid" / "images").glob("*")))
        n_test = len(list((root / "test" / "images").glob("*")))
        print(
            f"[kaggle_download] OK. train={n_train} valid={n_valid} test={n_test} images. "
            f"data.yaml found at {yaml_path}"
        )


def fix_data_yaml(root: Path):
    """Rewrite the train/val/test paths in data.yaml to be relative + correct,
    since many Roboflow exports ship broken absolute paths."""
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return
    lines = yaml_path.read_text().splitlines()
    # drop any pre-existing train/val/test keys, keep nc / names
    kept = [l for l in lines if not l.split(":")[0].strip() in ("train", "val", "test")]
    kept += [
        "train: train/images",
        "val: valid/images",
        "test: test/images",
    ]
    yaml_path.write_text("\n".join(kept) + "\n")
    print(f"[kaggle_download] Rewrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="kaggle dataset slug, e.g. user/dataset-name")
    parser.add_argument("--out", default="/content/data", help="output directory")
    parser.add_argument("--fix-yaml", action="store_true", help="rewrite data.yaml paths")
    args = parser.parse_args()

    root = download_dataset(args.dataset, args.out)
    verify_yolo_structure(root)
    if args.fix_yaml:
        fix_data_yaml(root)
