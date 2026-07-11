"""
Loads the dataset's COCO-format annotations (annotations/train_coco.json etc,
visible in your Kaggle input tree under COCO/COCO/annotations/) and rasterizes
polygon/RLE segmentations into per-pixel class masks for U-Net training.

This is the piece the original notebook never used - it only touched the
YOLO folder, even though the dataset ships proper segmentation masks in COCO.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class DentalSegDataset(Dataset):
    def __init__(self, images_dir: str, coco_json: str, img_size: int = 256, class_ids=None):
        self.images_dir = Path(images_dir)
        self.coco = COCO(coco_json)
        self.img_size = img_size
        # restrict to pathology classes only (skip pure-structural classes
        # like "Permanent Teeth" if you only care about disease segmentation)
        self.class_ids = class_ids or sorted(self.coco.getCatIds())
        self.cat_id_to_channel = {cid: i + 1 for i, cid in enumerate(self.class_ids)}  # 0 = background
        self.image_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.images_dir / img_info["file_name"]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        h, w = img.shape

        mask = np.zeros((h, w), dtype=np.int64)  # single-channel, value = class index
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.class_ids)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            channel = self.cat_id_to_channel.get(ann["category_id"])
            if channel is None:
                continue
            if isinstance(ann["segmentation"], list):
                rle = coco_mask.frPyObjects(ann["segmentation"], h, w)
                m = coco_mask.decode(coco_mask.merge(rle))
            else:
                m = coco_mask.decode(ann["segmentation"])
            mask[m.astype(bool)] = channel

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t

    @property
    def num_classes(self):
        return len(self.class_ids) + 1  # + background
