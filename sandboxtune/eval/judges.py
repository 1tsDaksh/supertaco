from typing import Dict, Any


def score_response(
    prompt: str, response: str, judge_model: str = "nvidia/nemotron-3-nano"
) -> float:
    """Use Nemotron judge to score a model response to a prompt.

    Returns a score from 0-10.
    """
    # In production, this would call the Nemotron judge model
    # For simulation, return a deterministic-ish score
    import hashlib
    hash_val = int(hashlib.md5(f"{prompt}{response}".encode()).hexdigest()[:8], 16)
    return round((hash_val % 80 + 30) / 10, 2)  # 3.0 to 11.0 range


def compare_models(
    base_responses: Dict[str, str],
    fine_tuned_responses: Dict[str, str],
    prompts: List[str],
) -> Dict[str, Any]:
    """Compare base vs fine-tuned model responses across prompts.

    Returns comparison dict with per-prompt scores and overall verdict.
    """
    results = {
        "prompt_results": [],
        "base_total": 0,
        "ft_total": 0,
        "regression": False,
    }

    for prompt in prompts:
        base_score = score_response(prompt, base_responses.get(prompt, ""))
        ft_score = score_response(prompt, fine_tuned_responses.get(prompt, ""))

        results["prompt_results"].append(
            {
                "prompt": prompt,
                "base_score": base_score,
                "fine_tuned_score": ft_score,
                "improvement": ft_score > base_score,
            }
        )
        results["base_total"] += base_score
        results["ft_total"] += ft_score

    if results["ft_total"] < results["base_total"] * 0.9:
        results["regression"] = True

    return results