"""Stitched HPSv3 value model: SD3.5 Medium latents scored by the HPSv3 reward model.

The trained adapter and vision suffix replace HPSv3's vision tower; the language model, prompt
template and reward head are the original HPSv3 ones (stitchvm/models/qwen2vl_reward.py).

- ``score``: images + prompts at a fixed noise level σ. Each image is resized to the VAE input
  size (1024² by default), encoded, noised as z_σ = (1 - σ) z + σ ε and scored; the stitched head
  runs on the image's native HPSv3 patch grid (the grid HPSv3 gives the original image).
- ``score_noisy_latents``: already-noised SD3.5 latents (e.g. inside a sampler) at noise level σ;
  the head grid is the HPSv3 grid of an image of the decoded size.
- ``score_original``: the unmodified HPSv3 model, for comparison.

Rewards are HPSv3's μ (the official inferencer output).
"""

import torch
from loguru import logger
from PIL import Image
from torchvision import transforms as T

from stitchvm.utils import extension_checkpoint as C
from stitchvm.utils.noisy_latent import make_noisy_latent


def _to_pil(image):
    return image if isinstance(image, Image.Image) else Image.open(image)


def build_stitched_hpsv3(ckpt: dict, reward_model, device):
    """StitchImage2D (SD3.5 Medium VAE + DiT prefix + trained adapter and suffix) for HPSv3."""
    from stitchvm.models.backbones.sd3 import SD3DiTStitched
    from stitchvm.models.feedforward.qwen2vl_vision_head import Qwen2VLVisionHead
    from stitchvm.models.stitched_image_model import StitchImage2D
    from stitchvm.models.stitching_layer_builder import ConvSpec
    from stitchvm.utils.utils_for_thirdparty import load_sd3_dit, load_sd3_prompt_embedding, load_sd3_vae

    enc_split, dit_split = C.split_location(ckpt["stitching_layer_location"])
    backbone = C.diffusion_backbone(ckpt)  # SD3.5 Medium at the pinned revision
    vae = load_sd3_vae(backbone, device).eval()
    head = Qwen2VLVisionHead.from_qwen2vl_visual(reward_model.visual, enc_split)
    head.to(torch.float32)
    dit_full = load_sd3_dit(backbone, device)
    dit = SD3DiTStitched(dit_full, dit_split).to(device).eval()
    del dit_full
    prompt_embedding = load_sd3_prompt_embedding(backbone, device)
    conv = ckpt["stitching_layer"]["stitching_conv.weight"]
    spec = ConvSpec(out_channels=int(conv.shape[0]), kernel_size=tuple(conv.shape[2:]), stride=1, padding=0)
    stitched = StitchImage2D(diffusion_vae=vae, clip_head=head, stitching_layer_config=spec, device=device, dit=dit)
    stitched.stitching_layer.load_state_dict(ckpt["stitching_layer"], strict=True)
    stitched.stitched_3d_model.load_state_dict(ckpt["stitched_3d_model"], strict=True)
    stitched.to(device).eval()
    for p in stitched.parameters():
        p.requires_grad = False
    return stitched, prompt_embedding


def check_hpsv3_identity(ckpt: dict, hpsv3_file: str, from_hub_cache: bool = False):
    """Portable checkpoints record the SHA-256 of the HPSv3 file they were trained with.

    A file given by the user is hashed in full. Only the pinned Hub download (from_hub_cache)
    uses the content-addressed blob name of the Hub cache instead of re-hashing ~16 GB. Training
    checkpoints carry no HPSv3 identity; for them the check is skipped with a warning."""
    meta = ckpt.get("export_meta")
    if meta is None:
        logger.warning("training checkpoint without export metadata: the HPSv3 file cannot be checked "
                       "against the one used for training")
        return
    if meta.get("extension") != "hpsv3":
        raise ValueError(f"checkpoint is for {meta.get('extension')!r}, not hpsv3")
    want = meta["reward_model"].get("file_sha256")
    got = C.file_sha256_cached(hpsv3_file) if from_hub_cache else C.sha256_file(hpsv3_file)
    if want and got != want:
        raise ValueError(f"{hpsv3_file} (sha256 {got}) is not the HPSv3 file this checkpoint was trained with ({want})")


class StitchedHPSv3:
    def __init__(self, reward_model, stitched, prompt_embedding, device, vae_input_size=1024):
        self.reward_model = reward_model
        self.stitched = stitched
        self.prompt_embeds, self.pooled_embeds = prompt_embedding
        self.device = device
        self.vae_input_size = vae_input_size
        self.vae_transform = T.Compose([
            T.Resize((vae_input_size, vae_input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_pretrained(cls, checkpoint: str, device="cuda", hpsv3_checkpoint=None, attn_implementation="sdpa",
                        vae_input_size=1024):
        """Load a portable (or training) checkpoint plus the pinned upstream models."""
        from stitchvm.models.qwen2vl_reward import Qwen2VLRewardModel, hpsv3_checkpoint_file

        ckpt = C.load_checkpoint(checkpoint)
        hpsv3_file = hpsv3_checkpoint or hpsv3_checkpoint_file()
        check_hpsv3_identity(ckpt, hpsv3_file, from_hub_cache=hpsv3_checkpoint is None)
        reward_model = Qwen2VLRewardModel.load_hpsv3(device, checkpoint_path=hpsv3_file,
                                                     attn_implementation=attn_implementation)
        stitched, prompt_embedding = build_stitched_hpsv3(ckpt, reward_model, device)
        logger.info(f"Stitched HPSv3 ready ({ckpt['stitching_layer_location']}, VAE input {vae_input_size}²)")
        return cls(reward_model, stitched, prompt_embedding, device, vae_input_size)

    @torch.no_grad()
    def vision_tokens(self, vae_images, grids, sigma: float, generator=None, chunk=10):
        """Merged vision tokens for VAE images in [-1, 1] (B, 3, S, S) at noise level sigma, each on its
        HPSv3 grid (t, h, w). Without a generator the global RNG is used (benchmark behaviour)."""
        merge = self.stitched.stitched_3d_model.spatial_merge_size
        embeds = []
        for i in range(0, len(vae_images), chunk):
            latent = self.stitched.vae_encoder_forward(vae_images[i:i + chunk].to(self.device))
            if sigma > 0 and generator is None:
                # the degenerate range (s, s) pins every sample to sigma = s
                noisy, actual = make_noisy_latent(latent, (sigma, sigma), distribution="uniform")
            elif sigma > 0:
                eps = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=torch.float32)
                noisy = (1 - sigma) * latent.float() + sigma * eps
                actual = torch.full((latent.shape[0],), float(sigma), device=latent.device)
            else:
                noisy, actual = latent, torch.zeros(latent.shape[0], device=latent.device)
            embeds += self._tokens_from_latent(noisy, actual, grids[i:i + len(latent)], merge)
        return torch.cat(embeds)

    def _tokens_from_latent(self, noisy, sigmas, grids, merge):
        out = []
        for b in range(noisy.shape[0]):
            _, gh, gw = grids[b]
            tokens = self.stitched(
                noisy_latent=noisy[b:b + 1], timestep=sigmas[b:b + 1] * 1000,
                encoder_hidden_states=self.prompt_embeds, pooled_projections=self.pooled_embeds,
                train=False, target_grid=(gh, gw),
            )
            assert tokens.shape[1] == gh * gw // (merge * merge)
            out.append(tokens[0])
        return out

    def _reward(self, inputs, vision_embeds=None):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.reward_model.rewards(self.reward_model(**inputs, vision_embeds=vision_embeds)).tolist()

    @torch.no_grad()
    def score(self, images, prompts, sigma: float = 0.0, seed: int = 0):
        """Stitched rewards for images (paths or PIL) and their prompts at noise level sigma."""
        images = [_to_pil(im).convert("RGB") for im in images]
        inputs, _ = self.reward_model.build_image_inputs(list(prompts), images)
        grids = inputs.pop("image_grid_thw").tolist()
        inputs.pop("pixel_values")
        vae_images = torch.stack([self.vae_transform(im) for im in images])
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return self._reward(inputs, self.vision_tokens(vae_images, grids, sigma, generator=gen))

    @torch.no_grad()
    def score_original(self, images, prompts):
        images = [_to_pil(im).convert("RGB") for im in images]
        inputs, _ = self.reward_model.build_image_inputs(list(prompts), images)
        return self._reward(inputs)

    @torch.no_grad()
    def score_noisy_latents(self, noisy_latents, sigma: float, prompts, image_size=None):
        """Rewards for normalized SD3.5 latents already noised to sigma (B, 16, H/8, W/8). image_size
        (height, width) of the decoded image defaults to 8x the latent size."""
        b, _, lh, lw = noisy_latents.shape
        height, width = image_size or (lh * 8, lw * 8)
        blank = [Image.new("RGB", (width, height)) for _ in range(b)]
        inputs, _ = self.reward_model.build_image_inputs(list(prompts), blank)
        grids = inputs.pop("image_grid_thw").tolist()
        inputs.pop("pixel_values")
        sigmas = torch.full((b,), float(sigma), device=self.device)
        merge = self.stitched.stitched_3d_model.spatial_merge_size
        tokens = self._tokens_from_latent(noisy_latents.to(self.device), sigmas, grids, merge)
        return self._reward(inputs, torch.cat(tokens))
