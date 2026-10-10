"""Prepare the AVA dataset for Stage 1 feature extraction.

AVA.txt (per-image rating histograms) is downloaded automatically; the images
(~32 GB) come from Kaggle:

    pip install kaggle  (and set up ~/.kaggle/kaggle.json)
    kaggle datasets download -d nicolacarrassi/ava-aesthetic-visual-assessment
    unzip ava-aesthetic-visual-assessment.zip
    # move the extracted JPEGs into <output_dir>/images/

    python data_processing/prepare_aesthetic_benchmark.py ava \\
        --output_dir data/aesthetic_benchmark/ava

Stage 1 uses --dataset "ava:<output_dir>" (all but the last 20K images).
"""

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

# AVA.txt is the per-image ratings file (~5 MB). We pull it from this
# GitHub mirror because it isn't in the Kaggle archive; the images
# (~32 GB) go through Kaggle.
AVA_TXT_URL = (
    "https://raw.githubusercontent.com/imfing/ava_downloader/"
    "master/AVA_dataset/AVA.txt"
)


def prepare_ava(args):
    """Download AVA.txt, then check the Kaggle images are in place."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    # Step 1: AVA.txt (ratings)
    ava_txt = output_dir / "AVA.txt"
    # An existing file is not necessarily a usable one: a truncated or
    # interrupted download leaves a short file that parses to nothing, so check
    # the line count rather than mere existence. AVA.txt has ~255,000 rows.
    MIN_ROWS = 200_000
    def _rows(path):
        try:
            with open(path) as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0
    if ava_txt.exists() and _rows(ava_txt) >= MIN_ROWS:
        print(f"AVA.txt already present at {ava_txt} ({_rows(ava_txt)} rows)")
    else:
        if ava_txt.exists():
            print(f"AVA.txt at {ava_txt} has only {_rows(ava_txt)} rows "
                  f"(expected >= {MIN_ROWS}) — re-downloading.")
        print("Downloading AVA.txt...")
        urlretrieve(AVA_TXT_URL, ava_txt)
        got = _rows(ava_txt)
        if got < MIN_ROWS:
            raise SystemExit(
                f"Downloaded AVA.txt has only {got} rows (expected >= {MIN_ROWS}). "
                f"The download may have been truncated; delete {ava_txt} and retry."
            )
        print(f"Saved to {ava_txt} ({got} rows)")

    # Step 2: parse ratings (sanity + image check below)
    entries = []
    with open(ava_txt) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            image_id = parts[1]
            ratings = [int(parts[i]) for i in range(2, 12)]
            if sum(ratings) == 0:
                continue
            entries.append(image_id)
    print(f"Parsed {len(entries)} entries from AVA.txt")
    if not entries:
        raise SystemExit(
            f"No usable entries parsed from {ava_txt}. Expected lines of the form "
            "'index image_id r1 ... r10 tag1 tag2 challenge_id'."
        )

    # Step 3: check the (Kaggle-downloaded) images are present.
    # Split convention: last 20K = test (eval), the rest = training.
    # Check the split used by Stage 1 with one directory enumeration.
    train_entries = entries[:-20000]
    if images_dir.exists():
        with os.scandir(images_dir) as it:
            present = {e.name for e in it}
        found = sum(1 for img_id in train_entries if f"{img_id}.jpg" in present)
        print(f"Found {found}/{len(train_entries)} training-split images in {images_dir}")
        if found < len(train_entries) * 0.5:
            _print_ava_download_instructions(images_dir)
    else:
        images_dir.mkdir(parents=True, exist_ok=True)
        _print_ava_download_instructions(images_dir)

    print(f"\n{'=' * 60}")
    print("AVA preparation complete!")
    print(f"{'=' * 60}")
    print(f"  AVA.txt:    {ava_txt}")
    print(f"  Images dir: {images_dir}")
    print("\nFor Stage 1:")
    print(f'  ... --dataset "ava:{output_dir}"')


def _print_ava_download_instructions(images_dir):
    print(f"\n{'=' * 60}")
    print("AVA images not found. Download via Kaggle:")
    print(f"{'=' * 60}")
    print()
    print("  pip install kaggle                                   # one-time")
    print("  # Set up ~/.kaggle/kaggle.json — see")
    print("  #   https://www.kaggle.com/docs/api")
    print("  kaggle datasets download -d nicolacarrassi/ava-aesthetic-visual-assessment")
    print("  unzip ava-aesthetic-visual-assessment.zip")
    print(f"  mv <extracted>/*.jpg {images_dir}/")
    print()
    print("Source: https://www.kaggle.com/datasets/nicolacarrassi/"
          "ava-aesthetic-visual-assessment")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the AVA dataset for Stage 1 feature extraction"
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    ava_parser = subparsers.add_parser("ava", help="Prepare AVA dataset")
    ava_parser.add_argument(
        "--output_dir", type=str, default="data/aesthetic_benchmark/ava"
    )

    args = parser.parse_args()
    prepare_ava(args)


if __name__ == "__main__":
    main()
