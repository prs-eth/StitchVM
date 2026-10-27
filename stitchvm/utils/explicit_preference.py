"""Conservative explicit-preference parser for saved pairwise-judge outputs.

This is an optional, offline alternative to the official UnifiedReward parsing
rule (stitchvm.models.qwen35_judge.parse_verdict: case-sensitive substrings
"Answer 1 is better" / "Answer 2 is better"). Scores computed with it must be
labelled "explicit-preference parser", never "official".

Policy (version ``explicit-preference-v1``):

- The parser sees the generated text only: no expected verdict, correctness
  label, model name or noise level. The same rule is applied to original and
  stitched outputs.
- Letter case and Markdown emphasis are normalized; fenced code blocks and
  blockquote lines are ignored.
- Recognized explicit candidate preferences: "Answer 1 is better" (also with
  "slightly / clearly / ..."), "Answer: 1 is better", "The better answer is
  Answer 1", "Better Answer: Answer 1", "I choose / prefer Answer 1", a final
  "decision / verdict: Answer 1" that ends the selection or is followed by a
  rationale, and a standalone final "Answer 1" line (including a boxed label)
  unless the sentence introducing it is an example, hypothetical or hedge.
  Answer 2 uses identical rules.
- Negated, hypothetical, quoted-instruction and question forms are ignored.
  Conflicting preferences return no verdict; an overall tie stated after the
  last preference returns no verdict.
- Bare answer letters or numbers ("D", "Answer: D", "Final answer: 1") never
  become a candidate preference. Descriptive comparisons alone are
  insufficient. Outputs are never repaired.

Dependency-free (standard library only).
"""

import re

VERSION = "explicit-preference-v1.1"


def normalize(text: str) -> str:
    """Lowercase; drop fenced code, blockquote lines and Markdown emphasis."""
    # Do not treat a quoted prompt or a fenced code example as a verdict.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = "\n".join(s for s in text.splitlines() if not s.lstrip().startswith(">"))
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return text.lower()


PATTERNS = (
    ("is_better", re.compile(
        r"\banswer\s*:?\s*([12])\s+is\s+(?:(?:the|clearly|slightly|marginally|significantly|much)\s+)?better\b")),
    ("better_answer_is", re.compile(
        r"\b(?:the\s+)?(?:better|preferred|best)\s+(?:answer|response)\s*(?:is\s*:?\s*|:\s*)[\"']?\s*answer\s*([12])\b")),
    ("explicit_preference", re.compile(r"\bi\s+(?:choose|select|prefer)\s+answer\s*([12])\b")),
    ("final_decision", re.compile(
        r"\b(?:final\s+)?(?:decision|verdict|choice|selection)\s*:\s*answer\s*([12])\b"
        r"(?=[ \t]*(?:[.!;,]|$|\n|because\b|as\b|\())")),
)
TIE = re.compile(
    r"\b(?:both\s+(?:answers|responses|answer\s*1\s+and\s+answer\s*2)\s+are\s+equally\s+(?:good|bad|poor)|"
    r"neither\s+(?:answer|response)\s+is\s+(?:clearly\s+)?better|it\s+is\s+a\s+tie)\b")
CONDITIONAL = re.compile(r"\b(?:if|whether|either|hypothetically|suppose|assuming|not|never)\b")
_REPORTED_SPEECH = re.compile(r"\b(?:say|says|said|write|output|phrase|format|example|claim|claimed|claims)\b")
# Extra hedges for the line introducing a standalone final label ("I might choose:").
_HEDGE = re.compile(r"\b(?:guess|might|perhaps|maybe|possibly)\b")


def _hypothetical_lead_in(line: str) -> bool:
    """True when the last sentence of the line before a standalone final label introduces an
    example, a hypothetical or a hedge ("The required format example is:", "If I had to guess,
    I might choose:") rather than an actual selection."""
    sentence = re.split(r"[.!?;]\s+", line.rstrip(" :.!?;"))[-1]
    return bool(CONDITIONAL.search(sentence) or _REPORTED_SPEECH.search(sentence) or _HEDGE.search(sentence))


def parse(text: str) -> dict:
    """Parse one generated judge output.

    Returns:
        {"verdict": 1 | 2 | None, "reason": str, "evidence": [...]}; reason is one of
        explicit_preference, conflicting_preferences, explicit_tie_or_no_preference,
        no_explicit_candidate_preference.
    """
    t = normalize(text)
    evidence = []
    for rule, pattern in PATTERNS:
        for m in pattern.finditer(t):
            # Sentence-local guards keep quoted/hypothetical/negated instructions out.
            start = max(t.rfind("\n", 0, m.start()), t.rfind(".", 0, m.start()),
                        t.rfind("!", 0, m.start()), t.rfind("?", 0, m.start())) + 1
            end_candidates = [v for v in (t.find("\n", m.end()), t.find(".", m.end()),
                                          t.find("!", m.end()), t.find("?", m.end())) if v >= 0]
            end = min(end_candidates, default=len(t))
            sentence = t[start:end]
            before = t[start:m.start()]
            suffix = t[m.end():end]
            if CONDITIONAL.search(before) or re.match(r"\s*(?:than\s+answer\s*[12]\s*)?(?:if|unless)\b", suffix):
                continue
            if (re.match(r"\s*(?:than\s+answer\s*[12]\s*)?\?", t[m.end():end + 1])
                    or _REPORTED_SPEECH.search(before)):
                continue
            if re.search(r"\b(?:not\s+true|false)\b", sentence):
                continue
            evidence.append({"label": int(m.group(1)), "rule": rule, "start": m.start(), "end": m.end(),
                             "span": t[max(0, m.start() - 45):min(len(t), m.end() + 75)]})
    # A standalone final candidate label is explicit; a bare A/D/1 is not.
    lines = [s.strip() for s in t.splitlines() if s.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^\\\[\s*|\s*\\\]$", "", last)
        last = re.sub(r"\\boxed\{\s*(answer\s*[12])\s*\}", r"\1", last)
        m = re.fullmatch(r"(?:final\s+answer\s*:\s*)?answer\s*([12])\s*[.!]?", last)
        if m and len(lines) > 1 and _hypothetical_lead_in(lines[-2]):
            m = None
        if m:
            evidence.append({"label": int(m.group(1)), "rule": "standalone_final_candidate",
                             "start": len(t) - len(lines[-1]), "end": len(t), "span": last})
    labels = {e["label"] for e in evidence}
    # Equal accuracy is not an overall tie; a later explicit choice may resolve an earlier
    # discussion of a tie. A later overall tie blocks a verdict.
    ties = list(TIE.finditer(t))
    tie = bool(ties) and (not evidence or max(m.end() for m in ties) > max(e["end"] for e in evidence))
    if tie:
        return {"verdict": None, "reason": "explicit_tie_or_no_preference", "evidence": evidence}
    if len(labels) > 1:
        return {"verdict": None, "reason": "conflicting_preferences", "evidence": evidence}
    if not labels:
        return {"verdict": None, "reason": "no_explicit_candidate_preference", "evidence": []}
    return {"verdict": labels.pop(), "reason": "explicit_preference", "evidence": evidence}
