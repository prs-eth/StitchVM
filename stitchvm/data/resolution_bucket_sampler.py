"""Resolution-homogeneous batches for feature extraction and distillation."""

import bisect
import random

from torch.utils.data import BatchSampler, ConcatDataset


class ResolutionBucketBatchSampler(BatchSampler):
    """Choose one resolution uniformly per batch and attach it to each index.

    Datasets receive ``(index, (width, height))``. The reward-encoder input
    stays at its native size; only the VAE input varies between batches.
    """

    def __init__(self, sampler, batch_size, resolutions, drop_last=True,
                 seed=None, world_size=1):
        super().__init__(sampler, batch_size, drop_last)
        self.resolutions = list(resolutions)
        if not self.resolutions:
            raise ValueError("resolutions must be non-empty")
        self.seed = seed
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.world_size > 1 and seed is None:
            raise ValueError("Distributed resolution sampling requires a shared seed")
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        # Stage 1 retains its unseeded, one-process random stream. Training
        # uses a separate seeded stream shared by ranks. Accelerate scatters
        # consecutive batches across ranks, so hold one bucket for that group.
        # Accumulation may combine micro-batches at different resolutions.
        rng = random if self.seed is None else random.Random(self.seed + self.epoch * 10_000_007)
        for batch_index, indices in enumerate(super().__iter__()):
            if batch_index % self.world_size == 0:
                resolution = rng.choice(self.resolutions)
            yield [(index, resolution) for index in indices]


class ConcatResolutionDataset(ConcatDataset):
    """Pass resolution-bearing indices through to concatenated datasets."""

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            return super().__getitem__(item)
        index, resolution = item
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index out of range: {index}")
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        offset = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
        return self.datasets[dataset_index][(index - offset, resolution)]
