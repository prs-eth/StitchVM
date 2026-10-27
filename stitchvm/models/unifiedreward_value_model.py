"""Stitched UnifiedReward-2.0 Qwen3.5 pairwise judge.

The checkpoint's language model generates the judgement as in the original model — the pinned
UnifiedReward pair-rank prompt, the checkpoint's chat template with enable_thinking=False, its
processor applied to the original image, greedy decoding — while the vision tower is replaced by
the stitched branch: the processor-resized image is rebuilt from pixel_values, encoded by the
SD3.5 Medium VAE, noised at a fixed σ with a fixed seed, and passed through the DiT prefix, the
trained adapter and the vision suffix on the image's native grid (stitchvm/models/qwen35_judge.py).

This is a *pairwise* judge for one image, one question and two candidate answers; it is not a
general visual question-answering interface.
"""

import torch
from loguru import logger
from PIL import Image

from stitchvm.utils import extension_checkpoint as C
from stitchvm.utils import explicit_preference as EP
from stitchvm.models import qwen35_judge as J
from stitchvm.utils.vlrb_protocol import ANSWER_1, ANSWER_2


def build_stitched_visual(trainables: str, checkpoint_dir: str, judge, sigma: float, seed: int, device):
    """Stitched vision branch from a training or portable checkpoint, refusing a checkpoint trained
    for another reward checkpoint (reward_meta identity)."""
    from stitchvm.models.backbones.sd3 import SD3DiTStitched
    from stitchvm.models.feedforward.qwen35_vision_head import (
        Qwen35VisionHead,
        assert_same_reward_checkpoint,
        load_qwen35_visual,
    )
    from stitchvm.models.stitched_image_model import StitchImage2D
    from stitchvm.models.stitching_layer_builder import ConvSpec
    from stitchvm.utils.utils_for_thirdparty import load_sd3_dit, load_sd3_prompt_embedding, load_sd3_vae

    ckpt = C.load_checkpoint(trainables)
    meta = ckpt.get("reward_meta")
    if meta is None:
        raise ValueError(f"{trainables} has no reward_meta: not a Qwen3.5 StitchVM checkpoint")
    visual, policy, cur = load_qwen35_visual(checkpoint_dir, device)
    assert_same_reward_checkpoint(meta, cur, f"stitched checkpoint {trainables}")
    location = ckpt["stitching_layer_location"]
    split, dit_split = C.split_location(location)
    head = Qwen35VisionHead.from_visual(visual, split).float()
    del visual
    backbone = C.diffusion_backbone(ckpt)  # SD3.5 Medium at the pinned revision
    dit_full = load_sd3_dit(backbone, device)
    dit = SD3DiTStitched(dit_full, dit_split)
    del dit_full  # free the full DiT before the text encoders load
    prompt_emb, pooled = load_sd3_prompt_embedding(backbone, device)
    vae = load_sd3_vae(backbone, device).eval()
    stitched = StitchImage2D(
        diffusion_vae=vae, clip_head=head, stitching_layer_config=ConvSpec(**meta["stitching_layer_config"]),
        device=device, dit=dit,
    )
    stitched.stitching_layer.load_state_dict(ckpt["stitching_layer"], strict=True)
    stitched.stitched_3d_model.load_state_dict(ckpt["stitched_3d_model"], strict=True)
    stitched.to(device).eval()
    for p in stitched.parameters():
        p.requires_grad = False
    return J.StitchedVisual(judge.original_visual, stitched, policy, (prompt_emb, pooled),
                            sigma=sigma, seed=seed), {"location": location, "reward_meta": meta}


def load_judge(checkpoint_dir: str, device, max_new_tokens: int = 4096):
    """The original UnifiedReward-2.0 Qwen3.5 judge (language model and vision tower)."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from stitchvm.models.feedforward.qwen35_vision_head import checkpoint_identity, require_identity

    require_identity(checkpoint_identity(checkpoint_dir), checkpoint_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(checkpoint_dir, local_files_only=True)
    return J.Qwen35Judge(model, processor, max_new_tokens=max_new_tokens)


class StitchedUnifiedRewardJudge:
    def __init__(self, judge, visual=None, sigma=None, seed=None):
        self.judge = judge
        self.visual = visual
        self.sigma, self.seed = sigma, seed

    @classmethod
    def from_pretrained(cls, checkpoint: str | None, reward_checkpoint: str, sigma: float = 0.0, seed: int = 0,
                        device="cuda", max_new_tokens: int = 4096):
        """checkpoint: portable (or training) StitchVM checkpoint; None runs the original judge."""
        judge = load_judge(reward_checkpoint, device, max_new_tokens)
        if checkpoint is None:
            return cls(judge)
        visual, info = build_stitched_visual(checkpoint, reward_checkpoint, judge, sigma, seed, device)
        logger.info(f"Stitched judge ready ({info['location']}, sigma={sigma}, seed={seed})")
        return cls(judge, visual, sigma, seed)

    @torch.no_grad()
    def judge_pair(self, image, question: str, answer1: str, answer2: str) -> dict:
        """Generate the judgement for one image, question and two candidate answers."""
        image = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
        inputs = self.judge.inputs(image, J.build_prompt(question, answer1, answer2))
        text, n_new = self.judge.generate(inputs, self.visual)
        has1, has2 = ANSWER_1 in text, ANSWER_2 in text
        return {
            "output": text,
            "new_tokens": n_new,
            # Official UnifiedReward phrases, exactly one present (the strict baseline parse).
            "verdict_official_phrases": 1 if (has1 and not has2) else 2 if (has2 and not has1) else None,
            "verdict_explicit_preference": EP.parse(text)["verdict"],
            "explicit_preference_version": EP.VERSION,
            "stitched": self.visual is not None, "sigma": self.sigma, "seed": self.seed,
            "vae_encode_device": getattr(self.visual, "last_vae_device", None),
        }
