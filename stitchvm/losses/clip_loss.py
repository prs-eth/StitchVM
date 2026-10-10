import torch.nn.functional as F


class LossCLIPStitched:
    """Distillation loss for CLIP stitching.

    Supervises both:
    - Image embedding (post visual_projection, used for contrastive scoring)
    - All tokens including CLS (dense per-token features)

    so the stitched model reproduces the exact same outputs as the
    original CLIP model.
    """

    def __init__(self, loss_type: str = "mse", token_weight: float = 1.0):
        if loss_type not in ("mse", "l1"):
            raise ValueError(f"loss_type must be 'mse' or 'l1', got {loss_type!r}")
        self.loss_type = loss_type
        self.token_weight = token_weight

    def _compute_loss(self, student, teacher):
        if self.loss_type == "mse":
            return F.mse_loss(student, teacher)
        return F.l1_loss(student, teacher)

    def __call__(self, student_output, teacher_output):
        """
        Args:
            student_output: (image_embeds, all_tokens) from CLIPEncoderHead
                image_embeds: (B, projection_dim) projected CLS embedding
                all_tokens: (B, 1+N, D) all hidden states including CLS at index 0
            teacher_output: CLIPVisionModelOutput from CLIPVisionModelWithProjection
                .image_embeds: (B, projection_dim) projected CLS embedding
                .last_hidden_state: (B, 1+N, D) all hidden states including CLS

        Returns:
            dict with "total_loss", "embedding_loss", "token_loss"
        """
        student_embeds, student_tokens = student_output
        teacher_embeds = teacher_output.image_embeds
        teacher_tokens = teacher_output.last_hidden_state

        # Image embedding loss in normalized (spherical) space
        student_embeds_norm = F.normalize(student_embeds.float(), dim=-1)
        teacher_embeds_norm = F.normalize(teacher_embeds.float(), dim=-1)
        embedding_loss = self._compute_loss(student_embeds_norm, teacher_embeds_norm)

        # All-token loss (CLS + patches)
        token_loss = self._compute_loss(
            student_tokens.float(), teacher_tokens.float()
        )

        total_loss = embedding_loss + self.token_weight * token_loss

        return {
            "total_loss": total_loss,
            "embedding_loss": embedding_loss,
            "token_loss": token_loss,
        }
