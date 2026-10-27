# Vendored subset of HPSv3

Source: https://github.com/MizzenAI/HPSv3 at commit
`bd0c5fcb5f587617b0169c07222ab78d01e2f3c2` (MIT, see LICENSE).

Only the two files StitchVM needs to reproduce HPSv3's native input processing
for **inference and evaluation** are vendored, byte-identical to upstream:

- `hpsv3/dataset/utils.py` — `process_vision_info` / `fetch_image` /
  `smart_resize` (image resizing to the 256 × 28 × 28 pixel budget).
- `hpsv3/dataset/data_collator_qwen.py` — the `INSTRUCTION` and
  `prompt_with_special_token` prompt text.

Upstream's `hpsv3/__init__.py`, which imports the HPSv3 training stack, is not
vendored, so `hpsv3` imports as a namespace package and installing the full
upstream repository is not required. The reward model itself is rebuilt in
`stitchvm/models/qwen2vl_reward.py` from the published checkpoint
`MizzenAI/HPSv3` (`HPSv3.safetensors`, revision `4f81e3e0`) on top of
`Qwen/Qwen2-VL-7B-Instruct` (revision `eed13092`).

Stitching-layer search and distillation do not use this package.

Install (inference and evaluation), from the repository root:

```bash
python -m pip install -r requirements-extensions.txt
```

This installs the package and `pandas`, which `data_collator_qwen.py` imports at
module level.
