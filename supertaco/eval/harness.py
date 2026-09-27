"""Fixed 5-prompt eval suite: template responses scored by real judge."""

from __future__ import annotations

from typing import Any

PROMPT_SUITE_SIZE = 5

DEFAULT_PROMPTS = [
    "Explain what LoRA fine-tuning is in two sentences.",
    "Write a Python function that loads a Hugging Face dataset.",
    "Summarize why gradient clipping prevents divergence.",
    "How does a chat template differ from a tokenizer?",
    "Give one reason eval scores can regress after training.",
]


def build_responses(final_config: dict) -> tuple[dict, dict]:
    """Deterministic base vs fine-tuned template responses.

    Fine-tuned quality depends on whether the final config is healthy, so a
    still-broken config produces visibly worse answers (spec 5.4).
    """
    from supertaco.agent.simlogs import classify_config_failure

    healthy = classify_config_failure(final_config) is None
    base = {
        p: f"Base model answer to '{p[:48]}': a generic, unstructured reply "
        f"without concrete details."
        for p in DEFAULT_PROMPTS
    }
    if healthy:
        fine_tuned = {
            p: f"Fine-tuned answer to '{p[:48]}': 1) precise structure "
            f"2) concrete details 3) correct terminology."
            for p in DEFAULT_PROMPTS
        }
    else:
        fine_tuned = {
            p: f"I'm not really sure about '{p[:48]}'... an uncertain, "
            f"rambling reply that hedges and repeats itself."
            for p in DEFAULT_PROMPTS
        }
    return base, fine_tuned


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
