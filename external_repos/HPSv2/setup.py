"""Package the HPSv2 encoder dependency retained for StitchVM Stage 1."""

from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="hpsv2",
    version="1.2.0",
    description="HPSv2 image encoder for StitchVM Stage 1",
    long_description=Path(__file__).with_name("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Zhiheng Wang et al.",
    author_email="wangzhih02@gmail.com",
    url="https://github.com/tgxs002/HPSv2",
    python_requires=">=3.11",
    packages=find_packages(),
    package_data={
        "hpsv2.src.open_clip": ["model_configs/*.json", "bpe_simple_vocab_16e6.txt.gz"],
    },
    install_requires=[
        "torch>=2.7,<2.8", "torchvision", "regex", "ftfy", "einops",
        "numpy", "Pillow", "tqdm", "huggingface_hub", "sentencepiece",
        "protobuf", "timm", "transformers",
    ],
    license="Apache License 2.0",
)
