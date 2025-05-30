# src/dataset.py

import os
import torch
from PIL import Image
from torchvision.datasets import CocoDetection


class ChocolateCocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, coco_json, transforms=None):
        self.img_folder = img_folder
        
        self.coco = CocoDetection(img_folder, coco_json, transforms=None)

        self.coco.loadImgs   = self.coco.coco.loadImgs
        self.coco.loadAnns   = self.coco.coco.loadAnns
        self.coco.getAnnIds  = self.coco.coco.getAnnIds
        
        self.ids = list(self.coco.ids)  # list of image IDs
        self.transforms = transforms

        cats = self.coco.coco.loadCats(self.coco.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.cat2label = {c['id']: i for i, c in enumerate(cats)}
        self.labels = [c['name'] for c in cats]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.coco.loadAnns(ann_ids)

        raw_name = self.coco.coco.loadImgs(img_id)[0]['file_name']
        fp = os.path.join(self.img_folder, raw_name)
        if not os.path.exists(fp):
            base, ext = os.path.splitext(raw_name)
            alt = base + ext.upper()
            alt_fp = os.path.join(self.img_folder, alt)
            if os.path.exists(alt_fp):
                fp = alt_fp
            else:
                raise FileNotFoundError(f"Could not find image {raw_name!r} or {alt!r} in {self.img_folder!r}")

        img = Image.open(fp).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[ann['category_id']] + 1)

        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
