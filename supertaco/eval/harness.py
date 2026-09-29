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

# Handcrafted per-prompt answers (spec D4: structured mock responses). Quality
# spread is deliberate: base ~mediocre, healthy ~expert, unhealthy ~rambling,
# so the real judge separates tiers instead of flooring everything at 0-1.
BASE_ANSWERS = {
    DEFAULT_PROMPTS[0]: (
        "LoRA is a method for fine-tuning large models. It updates only some "
        "parameters, which makes training cheaper and faster than full fine-tuning."
    ),
    DEFAULT_PROMPTS[1]: (
        "You use the datasets library. Call load_dataset with the dataset name, "
        "for example load_dataset('imdb'), and it downloads the data so you can "
        "train on it."
    ),
    DEFAULT_PROMPTS[2]: (
        "Gradient clipping caps the size of gradient updates. This keeps training "
        "stable when the learning rate is high or the gradients get large."
    ),
    DEFAULT_PROMPTS[3]: (
        "A tokenizer converts text into tokens the model can process. A chat "
        "template formats conversation turns into a single prompt, so both are "
        "part of preparing text for the model."
    ),
    DEFAULT_PROMPTS[4]: (
        "The model can overfit to the training data. It then performs worse on "
        "general evaluation prompts than before fine-tuning."
    ),
}

HEALTHY_ANSWERS = {
    DEFAULT_PROMPTS[0]: (
        "LoRA (Low-Rank Adaptation) freezes the pre-trained weights and learns "
        "small structured low-rank matrices A and B per layer, so the model is "
        "updated via W + BA instead of rewriting every weight. Because adaptation "
        "updates are empirically low-rank, this trains under 1% of the parameters "
        "while matching full fine-tuning quality at a fraction of the memory cost."
    ),
    DEFAULT_PROMPTS[1]: (
        "Use the datasets library, which downloads from the Hub and caches locally:\n\n"
        "from datasets import load_dataset\n\n\n"
        'def load_hf_dataset(name="imdb", split="train"):\n'
        "    ds = load_dataset(name, split=split)\n"
        '    print("Loaded", name, split, "rows:", len(ds))\n'
        "    return ds\n\n"
        "Requesting a split gives a ready Dataset object you can shuffle, slice, and map."
    ),
    DEFAULT_PROMPTS[2]: (
        "Backpropagation can produce a few enormous per-sample gradients that "
        "dominate the update and send the loss spiking. Gradient clipping rescales "
        "the global gradient norm to a threshold (e.g. 1.0) before the optimizer "
        "step, bounding the worst-case step size while leaving small, healthy "
        "gradients untouched — which prevents one bad batch from kicking training "
        "into divergence."
    ),
    DEFAULT_PROMPTS[3]: (
        "A tokenizer defines the model's alphabet: it splits raw text into "
        "vocabulary ids and back. A chat template is a formatting layer above "
        "that: it arranges role-tagged messages (system, user, assistant) with the "
        "right special tokens and an assistant-generation marker, so multi-turn "
        "conversations become one training-ready string. Tokenization answers "
        "'what are the units'; the template answers 'how is a conversation laid "
        "out before tokenization'."
    ),
    DEFAULT_PROMPTS[4]: (
        "Overfitting is the classic cause: training too long or at too high a "
        "learning rate on a narrow dataset makes the model memorize it and forget "
        "the broad capabilities of the base model — catastrophic forgetting. "
        "Held-out eval scores drop even though training loss looked fine, which is "
        "why an independent eval suite is needed to catch it."
    ),
}

UNHEALTHY_ANSWERS = {
    DEFAULT_PROMPTS[0]: (
        "LoRA is... hmm, some kind of fine-tuning? I think it changes a few "
        "layers, or maybe freezes them — I'm honestly uncertain. It might use "
        "matrices, and it could be cheaper than normal training, but I can't "
        "explain the mechanism and I'm not even sure it works."
    ),
    DEFAULT_PROMPTS[1]: (
        "To load a dataset from Hugging Face... I think you import something, "
        "maybe datasets? The function might be load_data or get_dataset or "
        "download_dataset — I'm not confident which. It probably needs the "
        "internet, and there may be a caching step, but I can't write the actual "
        "code."
    ),
    DEFAULT_PROMPTS[2]: (
        "Gradient clipping prevents divergence... maybe because gradients get too "
        "big? Or too small? I'm not sure which direction is the problem. It might "
        "be related to the learning rate, or to exploding gradients, but honestly "
        "I can't give a clear reason why clipping would help."
    ),
    DEFAULT_PROMPTS[3]: (
        "A chat template and a tokenizer... they sound kind of similar? Maybe the "
        "template is inside the tokenizer, or maybe it's the other way around. I "
        "think one of them converts text to numbers, but I'm not sure what the "
        "other one does or when you would use each."
    ),
    DEFAULT_PROMPTS[4]: (
        "Eval scores regress because... the data might be bad? Or the model "
        "overfits, or underfits? There are several possible causes and I don't "
        "know which one applies. It could be the learning rate, the number of "
        "epochs, or the dataset size, but I can't name the actual mechanism."
    ),
}


def build_responses(final_config: dict) -> tuple[dict, dict]:
    """Deterministic base vs fine-tuned handcrafted answers.

    Fine-tuned quality depends on whether the final config is healthy, so a
    still-broken config produces visibly worse answers (spec 5.4).
    """
    from supertaco.agent.simlogs import classify_config_failure

    healthy = classify_config_failure(final_config) is None
    base = {p: BASE_ANSWERS[p] for p in DEFAULT_PROMPTS}
    tier = HEALTHY_ANSWERS if healthy else UNHEALTHY_ANSWERS
    fine_tuned = {p: tier[p] for p in DEFAULT_PROMPTS}
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
