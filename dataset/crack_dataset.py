import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from dataset.transform import normalize_rgb_only, train_transform, val_transform

class ResizePad:
    def __init__(self, size=(448, 448), is_mask=False):
        self.size = size
        self.is_mask = is_mask

    def __call__(self, img):
        interp = Image.NEAREST if self.is_mask else Image.BILINEAR
        img = ImageOps.contain(img, self.size, method=interp)  # 保持比例缩放

        delta_w = self.size[0] - img.size[0]
        delta_h = self.size[1] - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = ImageOps.expand(img, padding, fill=0)
        return img

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(448, 448), mode='train', seed=None):
        self.size = size
        self.mode = mode
        self.seed = seed

        # 严格匹配文件名一致的图像和掩膜
        self.ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(mask_dir, os.path.splitext(f)[0] + '.png'))
        ])
        self.img_paths = [os.path.join(img_dir, f"{id}.jpg") for id in self.ids]
        self.mask_paths = [os.path.join(mask_dir, f"{id}.png") for id in self.ids]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.mode == 'train':
            img, mask = train_transform(img, mask, self.size)
        else:
            seed = self.seed + idx if self.seed is not None else None
            img, mask = val_transform(img, mask, self.size, seed=seed)

        return img, mask