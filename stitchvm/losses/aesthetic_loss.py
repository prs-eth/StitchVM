"""Distillation loss for aesthetic scorer stitching.

Three-level distillation:
  1. token_loss:     loss_fn on all hidden states (CLS + patches)
  2. embedding_loss: loss_fn on normalized embeddings (input to MLP)
  3. score_loss:     loss_fn on aesthetic scores (output of the frozen MLP)

The MLP is frozen and shared between the student and teacher score paths;
only the student's embedding carries gradients through it.
"""

import torch
import torch.nn.functional as F

_LOSS_FNS = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
}


class LossAestheticStitched:
    """Three-level distillation for aesthetic stitching.

    Loss = token_weight   * loss_fn(student_tokens, teacher_tokens)
         + embed_weight   * loss_fn(norm(student_embeds), norm(teacher_embeds))
         + score_weight   * loss_fn(MLP(norm(student)), MLP(norm(teacher)))

    Args:
        aesthetic_mlp: The frozen aesthetic MLP (shared with the student head).
        token_weight: Weight for raw token distillation.
        embed_weight: Weight for normalized embedding distillation.
        score_weight: Weight for aesthetic score distillation.
        loss_type: "mse" or "l1".
    """

    def __init__(
        self,
        aesthetic_mlp,
        *,
        token_weight: float = 1.0,
        embed_weight: float = 1.0,
        score_weight: float = 1.0,
        loss_type: str = "mse",
    ):
        if loss_type not in _LOSS_FNS:
            raise ValueError(
                f"loss_type must be one of {sorted(_LOSS_FNS)}, got {loss_type!r}"
            )
        self.token_weight = token_weight
        self.embed_weight = embed_weight
        self.score_weight = score_weight
        self.aesthetic_mlp = aesthetic_mlp
        self.loss_fn = _LOSS_FNS[loss_type]

    def __call__(self, student_output, teacher_output):
        student_embeds, student_tokens = student_output
        teacher_embeds = teacher_output.image_embeds
        teacher_tokens = teacher_output.last_hidden_state

        # ── 1. Token loss ──
        token_loss = self.loss_fn(student_tokens.float(), teacher_tokens.float())

        # ── 2. Embedding loss: on normalized embeddings ──
        student_norm = F.normalize(student_embeds.float(), dim=-1)
        with torch.no_grad():
            teacher_norm = F.normalize(teacher_embeds.float(), dim=-1)
        embedding_loss = self.loss_fn(student_norm, teacher_norm)

        # ── 3. Score loss: on aesthetic scores ──
        # Match the MLP's parameter dtype (it may run outside autocast).
        mlp_dtype = next(self.aesthetic_mlp.parameters()).dtype
        student_score = self.aesthetic_mlp(student_norm.to(mlp_dtype)).squeeze(-1)
        with torch.no_grad():
            teacher_score = self.aesthetic_mlp(teacher_norm.to(mlp_dtype)).squeeze(-1)
        score_loss = self.loss_fn(student_score.float(), teacher_score.float())

        total_loss = (
            self.token_weight * token_loss
            + self.embed_weight * embedding_loss
            + self.score_weight * score_loss
        )

        return {
            "total_loss": total_loss,
            "token_loss": token_loss,
            "embedding_loss": embedding_loss,
            "score_loss": score_loss,
            "student_aesthetic_score": student_score.detach().float().mean(),
            "teacher_aesthetic_score": teacher_score.detach().float().mean(),
        }
