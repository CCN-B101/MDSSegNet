import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def normalize_rgb_only(img, mask):
    img = TF.to_tensor(img)
    img = TF.normalize(
        img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    mask = TF.to_tensor(mask).squeeze(0).long()
    return img, mask


def train_transform(img, mask, size):
    """
    Training-only stochastic augmentation.
    Different independent-run seeds therefore produce different stochastic
    optimization trajectories while the underlying dataset split remains fixed.
    """
    img = TF.resize(img, size, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

    # Retained from the original pipeline.
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=size)
    img = TF.crop(img, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    angle = random.choice([0, 90, 180, 270])
    img = TF.rotate(img, angle)
    mask = TF.rotate(mask, angle)

    img = TF.adjust_brightness(
        img,
        brightness_factor=random.uniform(0.9, 1.1)
    )
    img = TF.adjust_contrast(
        img,
        contrast_factor=random.uniform(0.9, 1.1)
    )

    return normalize_rgb_only(img, mask)


def val_transform(img, mask, size, seed=None):
    """
    Deterministic validation/test preprocessing.

    Reviewer 4 statistical validation requires every independently trained
    model to be evaluated on exactly the same validation/test inputs.
    Therefore no random brightness/contrast perturbation is applied here.
    The seed argument is kept only for compatibility with CrackDataset.
    """
    img = TF.resize(img, size, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

    return normalize_rgb_only(img, mask)
