import argparse
import os

from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from stitchvm.data.resolution_bucket_sampler import (
    ConcatResolutionDataset,
    ResolutionBucketBatchSampler,
)


def _build_single_stitching_dataset(
    name: str,
    root_path: str,
    width: int,
    height: int,
    feedforward_width: int,
    feedforward_height: int,
    augmentation: bool = False,
):
    if name == "HPDv2":
        from stitchvm.data.hpdv2_datasets import HPDv2Dataset

        return HPDv2Dataset(
            root_path=root_path,
            width=width,
            height=height,
            feedforward_width=feedforward_width,
            feedforward_height=feedforward_height,
            split="train",
            color_augmentation=augmentation,
        )
    elif name == "ava":
        from stitchvm.data.ava_dataset import AVATrainingDataset

        return AVATrainingDataset(
            root_path=root_path,
            width=width,
            height=height,
            feedforward_width=feedforward_width,
            feedforward_height=feedforward_height,
            split="train",
            color_augmentation=augmentation,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def create_stitching_dataloader(
    datasets: list[tuple[str, str]],  # <── list of (name, root) pairs
    args: argparse.Namespace,
    augmentation: bool = False,
):
    logger.info("Creating dataloader")
    for n, r in datasets:
        logger.info(f"  Dataset: {n} at {r}")

    # Shared DataLoader worker settings. persistent_workers/prefetch_factor are
    # only valid when num_workers > 0, so build the kwargs conditionally.
    num_workers = getattr(args, "num_workers", 3)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=False)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 2)

    # build each dataset individually
    ds_objects = [
        _build_single_stitching_dataset(
            name=n,
            root_path=r,
            width=args.width,
            height=args.height,
            feedforward_width=args.feedforward_width,
            feedforward_height=args.feedforward_height,
            augmentation=augmentation,
        )
        for n, r in datasets
    ]

    buckets = getattr(args, "resolution_buckets", None)
    if buckets:
        train_set = (
            ds_objects[0] if len(ds_objects) == 1
            else ConcatResolutionDataset(ds_objects)
        )
        sampler = ResolutionBucketBatchSampler(
            RandomSampler(train_set), args.batch_size, buckets, drop_last=True,
            seed=getattr(args, "global_seed", None),
            world_size=int(os.environ.get("WORLD_SIZE", "1")) if hasattr(args, "global_seed") else 1,
        )
        loader = DataLoader(train_set, batch_sampler=sampler, **loader_kwargs)
        logger.info(
            f"DataLoader has {len(train_set)} samples from {len(ds_objects)} "
            f"dataset(s); batch size {args.batch_size}, resolution buckets={buckets}"
        )
        return loader, sampler

    train_set = ds_objects[0] if len(ds_objects) == 1 else ConcatDataset(ds_objects)
    sampler = None
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    logger.info(
        f"DataLoader has {len(train_set)} samples from "
        f"{len(ds_objects)} dataset(s); batch size {args.batch_size}, "
        f"resolution {args.width}×{args.height}"
    )

    return loader, sampler
