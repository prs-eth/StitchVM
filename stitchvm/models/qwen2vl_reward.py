"""HPSv3: a Qwen2-VL reward model with its language branch preserved (evaluation).

HPSv3 [MizzenAI/HPSv3] is Qwen/Qwen2-VL-7B-Instruct with a "<|Reward|>" special
token and a ranknet MLP head that outputs (mu, sigma); the reward is mu.

This module rebuilds it from the published checkpoint on the Transformers 5.x
stack and reproduces the official forward (hpsv3/model/qwen2vl_trainer.py,
`Qwen2VLRewardModelBT.forward`):

  * vision tokens are scattered into the token embeddings at the image pad
    positions;
  * the language model is called directly with inputs_embeds and NO position
    ids, i.e. plain sequential positions on all three M-RoPE sections (the
    official reward model bypasses get_rope_index);
  * the reward head runs in fp32 on every token and is pooled at the special
    token.

Inputs are built with the official message format, prompt text and image
processing (vendored, unmodified, in external_repos/HPSv3).

The stitched value model replaces only the vision tower output
(`vision_embeds`); everything downstream is this unchanged language branch.
Training never loads this module: Stage 1 and Stage 2 use only the vision tower
(stitchvm.utils.utils_for_thirdparty.load_hpsv3_visual).
"""

import os

import torch
import torch.nn as nn
from loguru import logger

HPSV3_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
HPSV3_BASE_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
HPSV3_REPO = "MizzenAI/HPSv3"
HPSV3_REVISION = "4f81e3e09edd82fe3c5f636444c721b592a735ca"
HPSV3_CHECKPOINT = "HPSv3.safetensors"
HPSV3_PIXELS = 256 * 28 * 28  # min_pixels = max_pixels in HPSv3RewardInferencer


def _hf_offline() -> bool:
    truthy = ("1", "true", "yes", "on")
    return any(
        os.environ.get(v, "").strip().lower() in truthy
        for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def hpsv3_checkpoint_file() -> str:
    """Local path of HPSv3.safetensors at the pinned Hub revision (downloaded if needed)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        HPSV3_REPO, HPSV3_CHECKPOINT, revision=HPSV3_REVISION, local_files_only=_hf_offline()
    )


def _remap_legacy_qwen2vl_keys(state_dict: dict) -> dict:
    """transformers 4.45 Qwen2VLForConditionalGeneration keys -> 5.x layout."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model.") and not k.startswith(("model.visual.", "model.language_model.")):
            k = "model.language_model." + k[len("model."):]
        out[k] = v
    return out


def _ranknet_head(hidden_size: int, output_dim: int) -> nn.Sequential:
    """HPSv3 rm_head_type="ranknet" without rm_head_kwargs (qwen2vl_trainer.py)."""
    return nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(1024, 16),
        nn.ReLU(),
        nn.Linear(16, output_dim),
    )


class Qwen2VLRewardModel(nn.Module):
    """HPSv3 reward model with a swappable vision branch."""

    def __init__(self, vlm, rm_head, processor, special_token_ids, pixels):
        super().__init__()
        self.vlm = vlm
        self.rm_head = rm_head
        self.processor = processor
        self.special_token_ids = list(special_token_ids)
        self.pixels = pixels

    @classmethod
    def load_hpsv3(cls, device="cuda", checkpoint_path=None, attn_implementation="sdpa"):
        import safetensors.torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        special_tokens = ["<|Reward|>"]
        processor = AutoProcessor.from_pretrained(
            HPSV3_BASE_MODEL, revision=HPSV3_BASE_REVISION, padding_side="right",
            local_files_only=_hf_offline(),
        )
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
        vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            HPSV3_BASE_MODEL,
            revision=HPSV3_BASE_REVISION,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            local_files_only=_hf_offline(),
        )
        vlm.resize_token_embeddings(len(processor.tokenizer))
        vlm.config.use_cache = False

        path = checkpoint_path or hpsv3_checkpoint_file()
        logger.info(f"Loading HPSv3 weights from {path}")
        sd = safetensors.torch.load_file(path, device="cpu")
        head_sd = {k[len("rm_head."):]: v for k, v in sd.items() if k.startswith("rm_head.")}
        vlm_sd = _remap_legacy_qwen2vl_keys({k: v for k, v in sd.items() if not k.startswith("rm_head.")})
        vlm.load_state_dict(vlm_sd, strict=True)
        rm_head = _ranknet_head(vlm.config.text_config.hidden_size, 2)
        rm_head.load_state_dict(head_sd, strict=True)
        rm_head.to(torch.float32)  # create_model_and_processor: model.rm_head.to(torch.float32)
        model = cls(vlm, rm_head, processor, special_ids, pixels=HPSV3_PIXELS)
        return model.to(device).eval()

    # ------------------------------------------------------------------ #
    # Inputs (official message format and image processing)              #
    # ------------------------------------------------------------------ #
    @property
    def visual(self):
        return self.vlm.model.visual

    def hpsv3_messages(self, prompts, images):
        from hpsv3.dataset.data_collator_qwen import INSTRUCTION, prompt_with_special_token

        px = self.pixels
        return [
            [{"role": "user", "content": [
                {"type": "image", "image": image, "min_pixels": px, "max_pixels": px},
                {"type": "text", "text": INSTRUCTION.format(text_prompt=text) + prompt_with_special_token},
            ]}]
            for text, image in zip(prompts, images)
        ]

    def build_image_inputs(self, prompts, images):
        """HPSv3RewardInferencer.prepare_batch: images are paths or PIL images."""
        from hpsv3.dataset.utils import process_vision_info

        messages = self.hpsv3_messages(prompts, images)
        image_inputs, _ = process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=image_inputs, padding=True, return_tensors="pt")
        return inputs, image_inputs

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None,
                vision_embeds=None, **_):
        """Official Qwen2VLRewardModelBT.forward. `vision_embeds` (B, N, D) replaces
        the vision tower output (stitched value model); otherwise pixels are
        encoded by the original tower."""
        cfg = self.vlm.config
        embed = self.vlm.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        token_id = cfg.image_token_id
        if vision_embeds is not None:
            vis = vision_embeds.reshape(-1, vision_embeds.shape[-1])
        elif pixel_values is not None:
            vis = self.visual(pixel_values.type(self.visual.get_dtype()), grid_thw=image_grid_thw).pooler_output
        else:
            vis = None
        if vis is not None:
            mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if int(mask[..., 0].sum()) != vis.shape[0]:
                raise ValueError(
                    f"{int(mask[..., 0].sum())} image placeholder tokens but {vis.shape[0]} vision embeddings"
                )
            inputs_embeds = inputs_embeds.masked_scatter(mask, vis.to(inputs_embeds.device, inputs_embeds.dtype))

        hidden = self.vlm.model.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state
        logits = self.rm_head(hidden.float())

        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in self.special_token_ids:
            special |= input_ids == tid
        B = input_ids.shape[0]
        n_special = len(self.special_token_ids)
        pooled = logits[special, ...].view(B, n_special, -1).view(B, -1)
        return pooled

    def rewards(self, pooled: torch.Tensor):
        """(mu, sigma) -> mu: "we select miu as the final output" (official inferencer)."""
        return pooled[:, 0].float()
