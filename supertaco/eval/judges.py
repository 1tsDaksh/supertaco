"""Nemotron judge scoring with deterministic hash fallback (spec 5.4)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

JUDGE_PROMPT = """You are a strict evaluation judge. Score the response from 0 to 10
(10 = excellent, 0 = unusable). Reply with a single number only.

Prompt: {prompt}
Response: {response}"""


@dataclass
class JudgeResult:
    score: float
    mode: str  # "real" | "fallback" (HTTP level) | "hash"
    error: Optional[str] = None


def _hash_score(prompt: str, response: str) -> float:
    digest = int(hashlib.sha256(f"{prompt}{response}".encode()).hexdigest()[:8], 16)
    return round(2.0 + (digest % 81) / 10, 2)  # 2.0 - 10.0


def _parse_score(text: str) -> Optional[float]:
    """Numerator of "8/10" wins; else the last number ("0 to 10" echoes lose)."""
    fraction = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:/|out of\s*)10\b", text)
    if fraction:
        return float(fraction.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)  # signed: "Score: -2" -> 0.0
    return float(numbers[-1]) if numbers else None


def score_response(
    prompt: str, response: str, llm=None, judge_model_key: str = "patch"
) -> JudgeResult:
    """Score one response. With an llm client: real Nemotron judge call.

    Parse failure retries the judge once (Nemotron occasionally emits empty
    content); a second failure or missing client falls back to the
    deterministic hash. HTTP-level mode (real/fallback) is inherited.
    """
    if llm is None:
        return JudgeResult(_hash_score(prompt, response), "hash", "no llm client")

    judge_prompt = JUDGE_PROMPT.format(prompt=prompt, response=response)
    text = ""
    for _ in range(2):  # one retry on empty/unparseable reply, then hash
        text = llm._call(
            judge_model_key,
            judge_prompt,
            max_tokens=512,  # Nemotron-3 reasons before answering; 16 -> empty content
            temperature=0.0,
        )
        http_mode = "real"
        http_error = None
        if llm.call_log:
            http_mode = llm.call_log[-1].get("mode", "real")
            http_error = llm.call_log[-1].get("error")
        raw = _parse_score(text)
        if raw is not None:
            score = max(0.0, min(10.0, raw))
            return JudgeResult(round(score, 2), http_mode, http_error)
    return JudgeResult(
        _hash_score(prompt, response), "hash", f"unparseable judge output: {text[:80]!r}"
    )


def compare_models(
    base_responses: dict, fine_tuned_responses: dict, prompts: list, llm=None
) -> dict:
    """Compare base vs fine-tuned responses across prompts."""
    results = {"prompt_results": [], "base_total": 0.0, "ft_total": 0.0, "regression": False}
    for prompt in prompts:
        base = score_response(prompt, base_responses.get(prompt, ""), llm=llm)
        ft = score_response(prompt, fine_tuned_responses.get(prompt, ""), llm=llm)
        results["prompt_results"].append(
            {
                "prompt": prompt,
                "base_score": base.score,
                "fine_tuned_score": ft.score,
                "improvement": ft.score > base.score,
            }
        )
        results["base_total"] += base.score
        results["ft_total"] += ft.score
    results["regression"] = results["ft_total"] < results["base_total"] * 0.9
    return results
