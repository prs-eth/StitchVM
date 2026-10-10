import os
import random

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class HPDv2Dataset(Dataset):
    """HPDv2 dataset that loads unique images directly from the extracted folder.

    HPDv2 has ~798K preference pairs but only ~430K unique images. Instead of
    going through HuggingFace pair indexing (which inflates to ~1.29M), this
    directly globs the image folder for one epoch = one pass over all ~430K
    unique images.

    Run ``data_processing/prepare_hpdv2_train.py`` once to fetch the train
    images, then pass ``--dataset HPDv2:/path/to/output_dir`` to training
    scripts. The dataset expects ``root_path/train/`` to contain the JPEGs
    (matches the layout produced by the prepare script); a flat layout where
    ``root_path/`` holds JPEGs directly also works.

    Args:
        root_path: Directory containing extracted HPDv2 images. Either
            ``root_path/{split}/<image>.jpg`` (extracted tarball layout) or
            ``root_path/<image>.jpg`` (flat).
        width, height: VAE input resolution.
        feedforward_width, feedforward_height: reward-model input resolution.
        split: ``train`` (only supported value for now).
        color_augmentation: Enable random crop + color jitter.
    """

    def __init__(
        self,
        root_path: str,
        width: int,
        height: int,
        feedforward_width: int,
        feedforward_height: int,
        split: str = "train",
        color_augmentation: bool = True,
    ):
        super().__init__()
        if root_path is None:
            raise ValueError(
                "HPDv2Dataset requires --dataset HPDv2:<path>. Run "
                "data_processing/prepare_hpdv2_train.py to fetch the images, "
                "then pass the output directory as the path."
            )
        self.width = width
        self.height = height
        self.feedforward_width = feedforward_width
        self.feedforward_height = feedforward_height
        self.color_augmentation = color_augmentation

        # Resolve image directory: prefer root_path/<split>/, fall back to root_path/
        candidate = os.path.join(root_path, split)
        image_dir = candidate if os.path.isdir(candidate) else root_path

        # Collect all image files
        self.image_paths = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )

        print(f"HPDv2 {split}: {len(self.image_paths)} unique images from {image_dir}")

        if self.color_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
            )

    def __len__(self):
        return len(self.image_paths)

    def get_item_sub(self, idx, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.color_augmentation:
            scale = random.uniform(1.1, 1.3)
            resize_h = int(height * scale)
            resize_w = int(width * scale)
            img_resized = TF.resize(img, (resize_h, resize_w))
            i, j, h, w = transforms.RandomCrop.get_params(
                img_resized, output_size=(height, width)
            )
            vae_tensor = TF.to_tensor(TF.crop(img_resized, i, j, h, w))
            vae_tensor = self.color_jitter(vae_tensor)
        else:
            vae_tensor = TF.to_tensor(TF.resize(img, (height, width)))

        # Feedforward resolution (same content, different size)
        feedforward_tensor = TF.resize(
            vae_tensor,
            (self.feedforward_height, self.feedforward_width),
        )

        # Normalize to [-1, 1]
        vae_tensor = vae_tensor * 2 - 1
        feedforward_tensor = feedforward_tensor * 2 - 1

        return dict(
            vae_image_tensor=vae_tensor,
            feedforward_image_tensor=feedforward_tensor,
            prompt="",
        )

    def __getitem__(self, item):
        # `item` is either a plain int (fixed-resolution training) or a
        # `(idx, (w, h))` tuple (multi-resolution via ResolutionBucketBatchSampler).
        if isinstance(item, tuple):
            idx, (w, h) = item
        else:
            idx, w, h = item, self.width, self.height
        # A handful of files can be corrupt: fall back to the next index
        # rather than crash, but give up after a few tries so a systematic
        # failure (unmounted disk, code bug) fails loudly instead of
        # spinning all dataloader workers forever.
        last_error = None
        for _ in range(10):
            try:
                return self.get_item_sub(idx, width=w, height=h)
            except Exception as e:
                last_error = e
                idx = (idx + 1) % len(self.image_paths)
        raise RuntimeError(
            f"Failed to load 10 consecutive images ending at "
            f"{self.image_paths[idx]}"
        ) from last_error
