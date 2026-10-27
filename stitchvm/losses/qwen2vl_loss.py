import torch.nn.functional as F


class LossQwen2VLMergedTokens:
    """Distillation loss on the post-merger image tokens (HPSv3, Qwen3.5).

    The PatchMerger output is exactly what the frozen language branch of the
    reward model consumes (it replaces the image placeholder tokens), so this
    loss supervises the stitched vision branch at that interface and nothing
    else: no pre-merger token term, no normalized-embedding term, and no
    language-model or reward-head forward.

    The loss is the mean squared error over every (token, channel) element of
    the merged sequence. Every token is valid: the teacher and the student
    share one image grid per batch, so there is no padding to mask.
    """

    def __call__(self, student_output, teacher_output):
        """
        Args:
            student_output: (merged_tokens, hidden_states) from the vision head
                with return_patch_tokens=True; only merged_tokens (B, N/4, D) is used.
            teacher_output: teacher wrapper output; .image_embeds is the native
                tower's merged tokens (B, N/4, D).

        Returns:
            dict with "total_loss" and "merged_token_loss" (the same tensor).
        """
        student_merged = student_output[0]
        teacher_merged = teacher_output.image_embeds
        if student_merged.shape != teacher_merged.shape:
            raise ValueError(
                f"student merged tokens {tuple(student_merged.shape)} != teacher "
                f"{tuple(teacher_merged.shape)}: the student grid must match the "
                "teacher's image grid"
            )
        loss = F.mse_loss(student_merged.float(), teacher_merged.float())
        return {"total_loss": loss, "merged_token_loss": loss}
