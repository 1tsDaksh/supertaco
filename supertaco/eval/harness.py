import json
from typing import List, Dict, Any


PROMPT_SUITE_SIZE = 5  # Fixed prompts per demo dataset


def run_eval_suite(
    endpoint_client,
    prompts: List[str],
    base_model_config: dict,
    fine_tuned_checkpoint: str,
) -> Dict[str, Any]:
    """Run the fixed prompt suite against base and fine-tuned models.

    Returns dict with per-prompt scores and overall evaluation.
    """
    results = {
        "prompts": [],
        "base_scores": [],
        "fine_tuned_scores": [],
        "regression_flagged": False,
    }

    for prompt in prompts[:PROMPT_SUITE_SIZE]:
        # Evaluate base model
        base_score = _query_endpoint(endpoint_client, prompt, base_model_config)
        # Evaluate fine-tuned model
        ft_score = _query_endpoint(endpoint_client, prompt, fine_tuned_checkpoint)

        results["prompts"].append(
            {"prompt": prompt, "base_score": base_score, "fine_tuned_score": ft_score}
        )
        results["base_scores"].append(base_score)
        results["fine_tuned_scores"].append(ft_score)

    # Check for regression
    if results["fine_tuned_scores"] and results["base_scores"]:
        avg_ft = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
        avg_base = sum(results["base_scores"]) / len(results["base_scores"])
        results["regression_flagged"] = avg_ft < avg_base * 0.9  # 10% tolerance

    return results


def _query_endpoint(
    endpoint_client, prompt: str, model_config: dict
) -> float:
    """Query an endpoint with a prompt and return a judge score (0-10)."""
    # In production, this would call the Nemotron judge
    # For simulation, return a reasonable score
    import random
    return round(random.uniform(3.0, 9.0), 2)


def check_regression(eval_results: Dict[str, Any]) -> bool:
    """Check if eval results show regression vs baseline.

    Returns True if regression is detected (flagged loudly).
    """
    return eval_results.get("regression_flagged", False)