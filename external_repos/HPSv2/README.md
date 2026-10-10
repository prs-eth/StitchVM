# HPSv2 encoder dependency

This directory contains the HPSv2 fork of OpenCLIP used to load the frozen
ViT-H/14 image encoder for StitchVM Stage 1.

- Upstream: [HPSv2](https://github.com/tgxs002/HPSv2), by Zhiheng Wang et al.
- Model-library origin: [OpenCLIP](https://github.com/mlfoundations/open_clip).
- Original HPSv2 license: [LICENSE](LICENSE). See also the repository [NOTICE](../../NOTICE).

The `hpsv2/src/open_clip/` implementation, model configurations, and tokenizer
vocabulary are retained. The package entry point and packaging have been narrowed
for this use; upstream training commands, benchmark entry points, example assets,
and notebooks are omitted. This package is an encoder dependency, not the full
upstream HPSv2 application.

Install from the repository root with:

```bash
python -m pip install -e ./external_repos/HPSv2
```

StitchVM uses `hpsv2.src.open_clip.create_model`. This namespace is separate from
the pip package `open_clip_torch`, which loads the DFN encoder.
