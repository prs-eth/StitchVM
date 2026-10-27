"""VL-RewardBench protocol pieces of the UnifiedReward pair-rank judge (standard library only).

Shared by evaluation (scripts/eval_unifiedreward_vlrb.py, via
stitchvm.models.qwen35_judge) and offline rescoring of saved outputs
(scripts/merge_vlrb_shards.py, scripts/summarize_vlrb.py), which therefore need
neither torch nor transformers.

- prompt: the UnifiedReward pair-rank image-understanding prompt, verbatim from
  github.com/CodeGoat24/UnifiedReward @ ff6f5202;
- answer order: the official script's random.seed(0), one random() draw per item in
  dataset order;
- categories: the official script's id-prefix mapping and a source-based mapping;
- parse_verdict: the official substring rule plus a strict rule (exactly one verdict
  string). These are the baseline parses; stitchvm.utils.explicit_preference is a
  separate, optional parser.
"""

import math
import random

PROMPT_SOURCE = (
    "github.com/CodeGoat24/UnifiedReward@ff6f520247a2d36e2aff7b2ba53a935afd437f66:"
    "inference_qwen/image_understanding/qwen_pair_rank_image_understanding.py"
)
ANSWER_1 = "Answer 1 is better"
ANSWER_2 = "Answer 2 is better"


def build_prompt(query: str, r1: str, r2: str) -> str:
    return (
        "You are given an image and a question related to it. Your job is to evaluate the two responses based on these five factors:\n\n"
        "1. Accuracy of Object Descriptions: Review how accurately the objects are described in the responses, ensuring they match those in the ground truth. Be mindful of irrelevant or incorrect objects being mentioned.\n\n"
        "2. Relationship Between Objects: Check if the response properly describes how the objects relate to each other, reflecting their actual positions or interactions, as seen in the image.\n\n"
        "3. Description of Attributes: Assess how well the response captures the attributes (e.g., size, color, shape) of the objects in the image, in line with the ground truth.\n\n"
        "4. Helpfulness: Consider whether the response offers useful information that enhances the understanding of the image. Does it add context or provide extra insights? Also, evaluate whether it follows the instructions given in the prompt.\n\n"
        "5. Ethical Concerns: Review the response to ensure it avoids sensitive, harmful, or inappropriate content. The response should be fair, respect privacy, and be free of bias or offensive material.\n\n"
        "After evaluating both answers, determine which one is better based on these factors and clearly state your decision, such as 'Answer 1 is better' or 'Answer 2 is better.'\n\n"
        f"Question: {query}\n"
        f"Answer 1: {r1}\n"
        f"Answer 2: {r2}\n"
    )


def official_order_draws(n: int, seed: int = 0) -> list[float]:
    """random.random() draws of the official script (random.seed(0), one per item, in order)."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]


def arrange(responses, human_ranking, draw: float):
    """(R1, R2, expected verdict) exactly as test_qwen_vlrewardbench.py."""
    if human_ranking[0] == 0:
        if draw < 0.5:
            return responses[0], responses[1], ANSWER_1
        return responses[1], responses[0], ANSWER_2
    if draw < 0.5:
        return responses[0], responses[1], ANSWER_2
    return responses[1], responses[0], ANSWER_1


def _id_prefix(id_value: str) -> str:
    split = min((id_value.find("_"), id_value.find("-")), key=lambda x: x if x != -1 else float("inf"))
    return id_value[:split] if split != -1 else id_value


def category_official_script(id_value: str) -> str:
    """Category of the official UnifiedReward script (id-prefix mapping; POVID ids fall to
    the default 'vlfeedback' -> general)."""
    dtype = {"RLAIF": "rlaif-v", "RLHF": "rlhf-v", "mathverse": "reasoning_tasks", "mmmu": "reasoning_tasks",
             "wildvision": "wildvision-battle"}.get(_id_prefix(id_value), "vlfeedback")
    return {"vlfeedback": "general", "povid": "hallucination", "reasoning_tasks": "reasoning",
            "rlhf-v": "hallucination", "rlaif-v": "hallucination", "wildvision-battle": "general"}[dtype]


def category_by_source(id_value: str) -> str:
    """Source-family category of the dataset card (POVID, RLAIF-V, RLHF-V -> hallucination;
    MMMU-Pro, MathVerse -> reasoning; everything else, e.g. WildVision -> general)."""
    p = _id_prefix(id_value)
    if p in ("hallucination", "RLAIF", "RLHF"):
        return "hallucination"
    if p in ("mathverse", "mmmu"):
        return "reasoning"
    return "general"


def parse_verdict(output: str, expected: str) -> dict:
    has1, has2 = ANSWER_1 in output, ANSWER_2 in output
    strict = 1 if (has1 and not has2) else 2 if (has2 and not has1) else None
    return {
        "official_correct": expected in output,
        "strict_verdict": strict,
        "strict_correct": strict is not None and (ANSWER_1 if strict == 1 else ANSWER_2) == expected,
        "invalid": strict is None,
        "both_present": has1 and has2,
    }


def summarize(records):
    """Official-rule and strict-rule accuracy, invalid count and per-category results under both
    category mappings (official script id-prefix mapping and source-based mapping)."""
    def acc(rs, key):
        return sum(r[key] for r in rs) / max(1, len(rs))

    s = {"n": len(records), "official_accuracy": acc(records, "official_correct"),
         "strict_accuracy": acc(records, "strict_correct"),
         "invalid_parses": sum(r["invalid"] for r in records),
         "both_verdicts_present": sum(r["both_present"] for r in records),
         "mean_new_tokens": sum(r["new_tokens"] for r in records) / max(1, len(records)),
         "seconds_per_item": sum(r["seconds"] for r in records) / max(1, len(records))}
    for mapping in ("category_official_script", "category_by_source"):
        cats = {}
        for r in records:
            cats.setdefault(r[mapping], []).append(r)
        per = {c: {"n": len(v), "official_accuracy": acc(v, "official_correct"),
                   "strict_accuracy": acc(v, "strict_correct"), "invalid": sum(x["invalid"] for x in v)}
               for c, v in sorted(cats.items())}
        s[mapping] = per
        if len(per) == 3:
            s[f"macro_official_{mapping}"] = sum(v["official_accuracy"] for v in per.values()) / 3
            s[f"macro_strict_{mapping}"] = sum(v["strict_accuracy"] for v in per.values()) / 3
    return s


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
