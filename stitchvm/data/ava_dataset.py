import os
import random

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

# Handle truncated images (common in AVA)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AVATrainingDataset(Dataset):
    """AVA (Aesthetic Visual Analysis) training dataset for stitching.

    Uses the standard AVA train split (all entries except the last 20,000).
    Returns the same dict format as HPDv2Dataset: vae_image_tensor,
    feedforward_image_tensor, prompt.

    Args:
        root_path: Path containing AVA images ({image_id}.jpg) and AVA.txt.
                   Expects root_path/images/ and root_path/AVA.txt,
                   OR pass ava_txt/ava_root explicitly.
        width, height: VAE input resolution.
        feedforward_width, feedforward_height: CLIP input resolution.
        split: "train" (first ~235K) or "test" (last 20K).
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
        self.width = width
        self.height = height
        self.feedforward_width = feedforward_width
        self.feedforward_height = feedforward_height
        self.color_augmentation = color_augmentation

        # Resolve paths: root_path should contain images/ and AVA.txt (or AVA_Files/AVA.txt)
        root_path = str(root_path)
        ava_root = os.path.join(root_path, "images")
        ava_txt = os.path.join(root_path, "AVA.txt")

        # Check common AVA directory layouts
        if not os.path.exists(ava_txt):
            alt = os.path.join(root_path, "AVA_Files", "AVA.txt")
            if os.path.exists(alt):
                ava_txt = alt

        if not os.path.exists(ava_root):
            ava_root = root_path

        # Parse AVA.txt
        entries = []
        with open(ava_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 12:
                    continue
                image_id = parts[1]
                ratings = [int(parts[i]) for i in range(2, 12)]
                total = sum(ratings)
                if total == 0:
                    continue
                entries.append(image_id)

        # Standard AVA split: last 20,000 for test
        if split == "test":
            entries = entries[-20000:]
        elif split == "train":
            entries = entries[:-20000]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        # Filter to existing images with a single directory enumeration.
        # A per-entry os.path.exists() here would issue ~250K stat round-trips
        # at every startup (per rank) — pathological on network/object storage;
        # one scandir LIST is what those filesystems are actually good at.
        with os.scandir(ava_root) as it:
            present = {e.name for e in it}
        self.samples = [
            os.path.join(ava_root, f"{image_id}.jpg")
            for image_id in entries
            if f"{image_id}.jpg" in present
        ]

        print(
            f"AVA {split}: {len(self.samples)} images "
            f"(filtered from {len(entries)} entries)"
        )

        if self.color_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
            )

    def __len__(self):
        return len(self.samples)

    def get_item_sub(self, idx, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")

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
            prompt="",  # AVA has no text prompts; field needed for compatibility
        )

    def __getitem__(self, item):
        # `item` is either a plain int (fixed-resolution training) or a
        # `(idx, (w, h))` tuple (multi-resolution via ResolutionBucketBatchSampler).
        if isinstance(item, tuple):
            idx, (w, h) = item
        else:
            idx, w, h = item, self.width, self.height
        # A handful of AVA files are corrupt: fall back to the next index
        # rather than crash, but give up after a few tries so a systematic
        # failure (unmounted disk, code bug) fails loudly instead of
        # spinning all dataloader workers forever.
        last_error = None
        for _ in range(10):
            try:
                return self.get_item_sub(idx, width=w, height=h)
            except Exception as e:
                last_error = e
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError(
            f"Failed to load 10 consecutive images ending at {self.samples[idx]}"
        ) from last_error
