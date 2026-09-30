"""Fixed 5-prompt eval suite: real VM responses scored by the judge."""

from __future__ import annotations

import json
from typing import Any

PROMPT_SUITE_SIZE = 5

DEFAULT_PROMPTS = [
    "Explain what LoRA fine-tuning is in two sentences.",
    "Write a Python function that loads a Hugging Face dataset.",
    "Summarize why gradient clipping prevents divergence.",
    "How does a chat template differ from a tokenizer?",
    "Give one reason eval scores can regress after training.",
]


def run_eval_suite(
    prompts, base_responses: dict, fine_tuned_responses: dict, llm: Any = None
) -> dict:
    """Run the fixed suite; every score comes from judges.score_response."""
    from supertaco.eval.judges import score_response

    results: dict[str, Any] = {
        "prompts": [],
        "base_scores": [],
        "fine_tuned_scores": [],
        "modes": [],
        "regression_flagged": False,
    }
    for prompt in prompts[:PROMPT_SUITE_SIZE]:
        base = score_response(prompt, base_responses[prompt], llm=llm)
        ft = score_response(prompt, fine_tuned_responses[prompt], llm=llm)
        results["prompts"].append(prompt)
        results["base_scores"].append(base.score)
        results["fine_tuned_scores"].append(ft.score)
        results["modes"].append(f"base:{base.mode} / ft:{ft.mode}")

    if results["fine_tuned_scores"] and results["base_scores"]:
        avg_ft = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
        avg_base = sum(results["base_scores"]) / len(results["base_scores"])
        results["regression_flagged"] = avg_ft < avg_base * 0.9
    return results


def check_regression(eval_results: dict) -> bool:
    """True when fine-tuned average is >10% below baseline."""
    return eval_results.get("regression_flagged", False)


RESPONSES_START = "###RESPONSES_JSON###"
RESPONSES_END = "###END_RESPONSES_JSON###"


def split_responses(log_text: str) -> tuple[str, dict | None]:
    """Strip the responses marker block from train logs and parse it.

    Tolerant by design: unterminated block, bad JSON, or a payload without
    string-valued `base`/`fine_tuned` dicts returns (logs, None); the block is
    still stripped when the markers are present and terminated.
    """
    start = log_text.find(RESPONSES_START)
    if start < 0:
        return log_text, None
    end = log_text.find(RESPONSES_END, start)
    train_logs = log_text
    if end < 0:
        return train_logs, None
    inner = log_text[start + len(RESPONSES_START) : end].strip()
    train_logs = (log_text[:start] + log_text[end + len(RESPONSES_END) :]).strip()
    try:
        payload = json.loads(inner)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return train_logs, None
    if not isinstance(payload, dict):
        return train_logs, None
    base, ft = payload.get("base"), payload.get("fine_tuned")
    if not isinstance(base, dict) or not isinstance(ft, dict):
        return train_logs, None
    if not base or not ft or not all(isinstance(v, str) for v in base.values()):
        return train_logs, None
    if not all(isinstance(v, str) for v in ft.values()):
        return train_logs, None
    return train_logs, payload
