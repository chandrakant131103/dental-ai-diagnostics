"""
Loads the dataset's COCO-format annotations (annotations/train_coco.json etc,
visible in your Kaggle input tree under COCO/COCO/annotations/) and rasterizes
polygon/RLE segmentations into per-pixel class masks for U-Net training.

FIX (train/inference mismatch): the previous version resized the *entire*
panoramic X-ray down to img_size x img_size and trained on that. But
api/main.py's run_pipeline() crops each detected tooth/finding box FIRST,
then resizes only that crop to img_size x img_size before calling the
segmenter. The model was trained on a completely different input
distribution (whole squeezed panoramic images, where a lesion is often
<0.1% of pixels) than what it sees at inference (a tight crop around one
finding, where the lesion is a large fraction of the crop). This is very
likely the primary cause of the dead/negative logits and 0% positive
Grad-CAM activation observed during debugging - the model was optimizing
against the wrong problem.

This version instead builds one training sample per annotation: crop a
padded box around that annotation (mirroring the padded YOLO-box crop used
at inference), and rasterize the mask *within that crop only*. This matches
what the segmenter will actually be asked to do in production.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class DentalSegDataset(Dataset):
    def __init__(self, images_dir: str, coco_json: str, img_size: int = 256,
                 class_ids=None, crop_padding: float = 0.25, min_crop: int = 32):
        """
        Args:
            crop_padding: fraction of the annotation's box size to pad on
                each side, so the segmenter sees some context around the
                finding rather than a tight, context-free crop. Should
                roughly match whatever padding the detector-box crop uses
                at inference (see api/main.py - currently 0, consider
                adding the same padding there too, see note in main.py).
            min_crop: minimum crop side length in source-image pixels,
                so tiny annotations don't get crops so small that resizing
                to img_size introduces heavy upsampling artifacts.
        """
        self.images_dir = Path(images_dir)
        self.coco = COCO(coco_json)
        self.img_size = img_size
        self.crop_padding = crop_padding
        self.min_crop = min_crop
        # restrict to pathology classes only (skip pure-structural classes
        # like "Permanent Teeth" if you only care about disease segmentation)
        self.class_ids = class_ids or sorted(self.coco.getCatIds())
        self.cat_id_to_channel = {cid: i + 1 for i, cid in enumerate(self.class_ids)}  # 0 = background
        self.image_ids = sorted(self.coco.getImgIds())

        # Build one (image, anchor_annotation) sample per annotation whose
        # category is in class_ids. The anchor annotation defines the crop;
        # any other annotations whose box overlaps that crop are rasterized
        # into the mask too, since a real tooth crop can contain more than
        # one finding (e.g. Filling + Caries in the same tooth).
        self.samples = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids)
            for ann_id in ann_ids:
                self.samples.append((img_id, ann_id))

        if not self.samples:
            raise ValueError(
                "DentalSegDataset found zero annotations matching class_ids "
                f"{self.class_ids} in {coco_json}. Check the COCO json actually "
                "has segmentation fields (not just empty lists) - see "
                "data/yolo_to_coco.py output."
            )

    def __len__(self):
        return len(self.samples)

    def _anns_to_mask(self, anns, h, w):
        mask = np.zeros((h, w), dtype=np.int64)
        for ann in anns:
            channel = self.cat_id_to_channel.get(ann["category_id"])
            if channel is None:
                continue
            seg = ann.get("segmentation")
            if not seg:
                continue
            if isinstance(seg, list):
                if len(seg) == 0 or len(seg[0]) < 6:
                    continue  # degenerate polygon, skip rather than crash
                rle = coco_mask.frPyObjects(seg, h, w)
                m = coco_mask.decode(coco_mask.merge(rle))
            else:
                m = coco_mask.decode(seg)
            mask[m.astype(bool)] = channel
        return mask

    def __getitem__(self, idx):
        img_id, anchor_ann_id = self.samples[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.images_dir / img_info["file_name"]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        h, w = img.shape

        anchor = self.coco.loadAnns([anchor_ann_id])[0]
        ax, ay, aw, ah = anchor["bbox"]  # COCO bbox format: x, y, width, height

        pad_x = aw * self.crop_padding
        pad_y = ah * self.crop_padding
        x1 = int(max(0, ax - pad_x))
        y1 = int(max(0, ay - pad_y))
        x2 = int(min(w, ax + aw + pad_x))
        y2 = int(min(h, ay + ah + pad_y))

        # guard against degenerate / too-small crops
        if x2 - x1 < self.min_crop:
            cx = (x1 + x2) // 2
            x1, x2 = max(0, cx - self.min_crop // 2), min(w, cx + self.min_crop // 2)
        if y2 - y1 < self.min_crop:
            cy = (y1 + y2) // 2
            y1, y2 = max(0, cy - self.min_crop // 2), min(h, cy + self.min_crop // 2)

        img_crop = img[y1:y2, x1:x2]

        # find every annotation on this image whose box overlaps the crop,
        # so multi-finding teeth get a complete mask, not just the anchor
        all_ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids)
        all_anns = self.coco.loadAnns(all_ann_ids)
        overlapping = []
        for ann in all_anns:
            bx, by, bw, bh = ann["bbox"]
            if bx < x2 and bx + bw > x1 and by < y2 and by + bh > y1:
                overlapping.append(ann)

        full_mask = self._anns_to_mask(overlapping, h, w)
        mask_crop = full_mask[y1:y2, x1:x2]

        img_resized = cv2.resize(img_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img_resized).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask_resized).long()
        return img_t, mask_t

    @property
    def num_classes(self):
        return len(self.class_ids) + 1  # + background

    def compute_class_pixel_weights(self, max_samples: int = 500) -> torch.Tensor:
        """Inverse-frequency pixel weights per class (incl. background),
        estimated from a subsample so this stays fast on large datasets.
        Feed straight into nn.CrossEntropyLoss(weight=...).
        """
        counts = np.zeros(self.num_classes, dtype=np.int64)
        n = min(max_samples, len(self))
        idxs = np.linspace(0, len(self) - 1, n).astype(int)
        for i in idxs:
            _, mask_t = self[i]
            vals, cnts = np.unique(mask_t.numpy(), return_counts=True)
            for v, c in zip(vals, cnts):
                if 0 <= v < self.num_classes:
                    counts[v] += c
        counts = np.maximum(counts, 1)  # avoid div-by-zero for classes absent in the subsample
        freq = counts / counts.sum()
        weights = 1.0 / freq
        weights = weights / weights.sum() * self.num_classes  # normalize so mean weight ~1
        return torch.tensor(weights, dtype=torch.float32)
