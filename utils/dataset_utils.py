import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# === ROOT DIRECTORIES (set once globally) ===
ROOT = "F:/test_yolo_crack_detection/data/raw/rdd2022/datasets"
TRAIN_IMG_DIR = f"{ROOT}/train/images"
VAL_IMG_DIR   = f"{ROOT}/valid/images"
TEST_IMG_DIR  = f"{ROOT}/test/images"

TRAIN_LBL_DIR = f"{ROOT}/train/labels"
VAL_LBL_DIR   = f"{ROOT}/valid/labels"
TEST_LBL_DIR  = f"{ROOT}/test/labels"


class CrackDataset(Dataset):
    """
    Absolute-path robust loader for RDD2022 dataset.
    - Auto-detects correct split (train/valid/test)
    - Falls back between label dirs if needed
    - Skips blank/empty-label samples safely
    """

    def __init__(self, list_path, img_size=640, debug=False):
        self.img_size = img_size
        self.debug = debug

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"❌ Dataset list not found: {list_path}")

        with open(list_path, "r") as f:
            self.img_paths = [
                x.strip().replace("\\", "/")
                for x in f.read().splitlines()
                if x.strip()
            ]

        if len(self.img_paths) == 0:
            raise ValueError(f"❌ No image paths found in {list_path}")

        if self.debug:
            print(f"🧩 Loaded {len(self.img_paths)} image paths from {list_path}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index].replace("\\", "/")

        # Detect split to find label directory
        if "/train/" in img_path:
            label_dir = TRAIN_LBL_DIR
        elif "/valid/" in img_path or "/val/" in img_path:
            label_dir = VAL_LBL_DIR
        elif "/test/" in img_path:
            label_dir = TEST_LBL_DIR
        else:
            label_dir = TEST_LBL_DIR  # fallback

        base_name = os.path.basename(img_path).rsplit(".", 1)[0] + ".txt"
        label_path = f"{label_dir}/{base_name}"

        # Fallback chain if missing
        if not os.path.exists(label_path):
            for alt_dir in [TRAIN_LBL_DIR, VAL_LBL_DIR, TEST_LBL_DIR]:
                alt_path = f"{alt_dir}/{base_name}"
                if os.path.exists(alt_path):
                    label_path = alt_path
                    break

        # ---- Load image ----
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"❌ Image not loaded: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]

        # ---- Load labels ----
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cls, x, y, w, h = map(float, parts)
                            # Validate normalized coordinates
                            if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                                boxes.append([x, y, w, h, int(cls)])
                        except Exception:
                            continue
        else:
            if self.debug:
                print(f"⚠️ Missing label file: {label_path}")

        boxes = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 5), dtype=torch.float32)
        )

        # ✅ Skip blank labels (auto-fetch next valid image)
        if boxes.shape[0] == 0:
            return self.__getitem__((index + 1) % len(self))

        if self.debug and index < 3:
            print(f"📸 {os.path.basename(img_path)} → Labels: {boxes.shape[0]}")

        return img, boxes
