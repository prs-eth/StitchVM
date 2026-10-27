"""Image scoring with the twelve CLIP, HPSv2, Aesthetic and DFN-CLIP StitchVM recipes."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from stitchvm.models.stitched_image_model import StitchImage2D
from stitchvm.models.stitching_layer_builder import ConvSpec
from stitchvm.utils import utils_for_thirdparty as loaders


# One-based block outputs, matching the published training recipes.
LAYER_PAIRS = {
    "sd3.5": {"hpsv2": (3, 1), "clip": (3, 2), "aesthetic": (3, 2), "dfn_clip": (5, 2)},
    "sd3.5_large": {"hpsv2": (13, 1), "clip": (12, 2), "aesthetic": (12, 2), "dfn_clip": (15, 2)},
    "flux_dev": {name: (9, 2 if name != "hpsv2" else 1)
                 for name in ("hpsv2", "clip", "aesthetic", "dfn_clip")},
}


class StitchedImageReward:
    """Return cosine similarity, or the frozen Aesthetic MLP score, from noisy latents.

    The diffusion backbone always receives the empty prompt used during training.
    The user prompt is encoded only by the reward model's original text encoder.
    """

    def __init__(self, model, stitched, prompt_embedding, text_model, tokenizer, device, size):
        self.model = model
        self.stitched = stitched
        self.prompt_embedding = prompt_embedding
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    @classmethod
    def from_pretrained(cls, checkpoint, model="clip", backbone="sd3.5", device="cuda",
                        vae_input_size=1024, hpsv2_checkpoint_dir=None, aesthetic_mlp_path=None):
        from stitchvm.models.backbones.sd3 import SD3DiTStitched
        from stitchvm.models.backbones.flux import FluxDiTStitched
        from stitchvm.models.feedforward.clip_head import CLIPEncoderHead
        from stitchvm.models.feedforward.aesthetic_head import AestheticEncoderHead
        from stitchvm.models.feedforward.hpsv2_head import HPSv2EncoderHead
        from stitchvm.models.feedforward.dfn_clip_head import DFNClipEncoderHead

        if backbone not in LAYER_PAIRS or model not in LAYER_PAIRS[backbone]:
            raise ValueError(f"Unsupported backbone/reward pair: {backbone}/{model}")
        if vae_input_size <= 0 or vae_input_size % 16:
            raise ValueError("vae_input_size must be a positive multiple of 16")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ckpt.get("other_trainables"):
            raise ValueError("This loader expects the released adapter/suffix-only checkpoints")
        dit_split, enc_split = LAYER_PAIRS[backbone][model]
        location = f"enc_blocks_{enc_split}_dit_{dit_split}"
        if ckpt.get("stitching_layer_location", location) != location:
            raise ValueError(f"Checkpoint layer pair does not match {backbone}/{model}: expected {location}")
        if ckpt.get("export_meta") or ckpt.get("reward_meta"):
            raise ValueError("Use the HPSv3 or UnifiedReward loader for vision-language checkpoints")
        args = SimpleNamespace(diffusion_model=backbone, hpsv2_checkpoint_dir=hpsv2_checkpoint_dir)
        text_model = tokenizer = None
        # Build the reward suffix on CPU, then release the unused vision prefix.
        if model in ("clip", "aesthetic"):
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
            args.clip_model_id = ("openai/clip-vit-large-patch14" if model == "aesthetic"
                                  else "openai/clip-vit-large-patch14-336")
            vision, _ = loaders.load_clip_model(args, "cpu")
            if model == "aesthetic":
                if not aesthetic_mlp_path:
                    raise ValueError("aesthetic_mlp_path is required; run scripts/download_weights.py aesthetic")
                head = AestheticEncoderHead.from_clip_vision_model(vision, enc_split, aesthetic_mlp_path)
            else:
                head = CLIPEncoderHead.from_clip_vision_model(vision, enc_split)
                text_model = CLIPTextModelWithProjection.from_pretrained(
                    args.clip_model_id, local_files_only=loaders._hf_offline()).eval()
                tokenizer = AutoTokenizer.from_pretrained(args.clip_model_id, local_files_only=loaders._hf_offline())
            del vision
        else:
            if model == "hpsv2":
                if not hpsv2_checkpoint_dir:
                    raise ValueError("hpsv2_checkpoint_dir is required; run scripts/download_weights.py hpsv2")
                from hpsv2.src.open_clip import get_tokenizer
                text_model, _ = loaders.load_hpsv2_model(args, "cpu")
                head = HPSv2EncoderHead.from_hpsv2_model(text_model, enc_split)
                tokenizer = get_tokenizer("ViT-H-14")
            else:
                import open_clip
                text_model, _ = loaders.load_dfn_clip_model(args, "cpu")
                head = DFNClipEncoderHead.from_dfn_clip_model(text_model, enc_split)
                tokenizer = open_clip.get_tokenizer("hf-hub:apple/DFN5B-CLIP-ViT-H-14-378")
            del text_model.visual
        head.load_state_dict(ckpt["stitched_3d_model"], strict=True)
        # Encode the empty diffusion prompt before loading the DiT to reduce peak memory.
        is_flux = backbone == "flux_dev"
        prompt = (loaders.load_flux_prompt_embedding if is_flux else loaders.load_sd3_prompt_embedding)(args, device)
        full_dit = (loaders.load_flux_dit if is_flux else loaders.load_sd3_dit)(args, device)
        dit = FluxDiTStitched(full_dit, dit_split, guidance_scale=3.5) if is_flux else SD3DiTStitched(full_dit, dit_split)
        del full_dit
        vae = (loaders.load_flux_vae if is_flux else loaders.load_sd3_vae)(args, device).eval()
        stitched = StitchImage2D(vae, head, ConvSpec(head.hidden_size, (1, 1)), device=device, dit=dit)
        stitched.stitching_layer.load_state_dict(ckpt["stitching_layer"], strict=True)
        stitched.to(device).eval().requires_grad_(False)
        if text_model is not None:
            text_model.to(device).eval().requires_grad_(False)
        return cls(model, stitched, prompt, text_model, tokenizer, device, vae_input_size)

    @torch.inference_mode()
    def score(self, images, prompt=None, sigma=0.0, seed=0):
        if not 0 <= sigma <= 1:
            raise ValueError("sigma must be in [0, 1]")
        if self.model != "aesthetic" and not prompt:
            raise ValueError("A prompt is required for text-image similarity")
        text = None
        if self.model == "clip":
            tokens = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
            text = self.text_model(**tokens).text_embeds
        elif self.model != "aesthetic":
            text = self.text_model.encode_text(self.tokenizer([prompt]).to(self.device))
        if text is not None:
            text = F.normalize(text.float(), dim=-1)
        scores = []
        for image in images:
            if isinstance(image, Image.Image):
                rgb = image.convert("RGB")
            else:
                with Image.open(image) as opened:
                    rgb = opened.convert("RGB")
            latent = self.stitched.vae_encoder_forward(self.transform(rgb).unsqueeze(0).to(self.device))
            # Reset per image: changing the list order does not change its noise draw.
            generator = torch.Generator(device=self.device).manual_seed(seed)
            eps = torch.randn(latent.shape, generator=generator, device=self.device, dtype=torch.float32)
            noisy = (1 - sigma) * latent.float() + sigma * eps
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                embed = self.stitched(
                    noisy_latent=noisy, timestep=torch.tensor([sigma * 1000], device=self.device),
                    encoder_hidden_states=self.prompt_embedding[0], pooled_projections=self.prompt_embedding[1])
            if self.model == "aesthetic":
                value = self.stitched.stitched_3d_model.compute_aesthetic_score(embed)
            else:
                value = (F.normalize(embed.float(), dim=-1) * text).sum(-1)
            scores.append(float(value.item()))
        return scores
