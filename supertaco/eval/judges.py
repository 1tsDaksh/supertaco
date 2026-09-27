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


def score_response(
    prompt: str, response: str, llm=None, judge_model_key: str = "patch"
) -> JudgeResult:
    """Score one response. With an llm client: real Nemotron judge call.

    Parse failure or missing client falls back to the deterministic hash;
    the HTTP-level mode (real/fallback) is inherited from the client.
    """
    if llm is None:
        return JudgeResult(_hash_score(prompt, response), "hash", "no llm client")

    text = llm._call(
        judge_model_key,
        JUDGE_PROMPT.format(prompt=prompt, response=response),
        max_tokens=16,
        temperature=0.0,
    )
    http_mode = "real"
    http_error = None
    if llm.call_log:
        http_mode = llm.call_log[-1].get("mode", "real")
        http_error = llm.call_log[-1].get("error")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)  # signed so "Score: -2" clamps to 0.0
    if not numbers:
        return JudgeResult(
            _hash_score(prompt, response), "hash", f"unparseable judge output: {text[:80]!r}"
        )
    raw = float(numbers[-1])  # last number: avoids "0 to 10" echoes
    score = max(0.0, min(10.0, raw))
    return JudgeResult(round(score, 2), http_mode, http_error)


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
