"""Real judge scoring with hash fallback, clamping, and suite regression flag."""

from supertaco.eval.harness import DEFAULT_PROMPTS, build_responses, run_eval_suite
from supertaco.eval.judges import JudgeResult, score_response


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.call_log = []

    def _call(self, model_key, prompt, max_tokens=1024, temperature=0.0):
        text = self.replies.pop(0)
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.1, "error": None}
        )
        return text


def test_hash_fallback_without_client():
    result = score_response("p", "r")
    assert isinstance(result, JudgeResult)
    assert result.mode == "hash"
    assert 2.0 <= result.score <= 10.0


def test_real_judge_score_parsed_and_clamped():
    llm = FakeLLM(["8"])  # first prompt, base
    result = score_response("p", "r", llm=llm)
    assert result.score == 8.0
    assert result.mode == "real"


def test_clamp_out_of_range():
    llm = FakeLLM(["15", "Score: -2"])
    assert score_response("p", "r", llm=llm).score == 10.0
    assert score_response("p", "r", llm=llm).score == 0.0


def test_unparseable_judge_output_falls_back_to_hash():
    llm = FakeLLM(["no numbers here"])
    result = score_response("p", "r", llm=llm)
    assert result.mode == "hash"
    assert 2.0 <= result.score <= 10.0


def test_judge_requests_enough_tokens_for_reasoning_model():
    seen = {}

    class _Recorder:
        call_log = []

        def _call(self, model_key, prompt, max_tokens=1024, temperature=0.0):
            seen["max_tokens"] = max_tokens
            return "8/10"

    score_response("p", "r", llm=_Recorder())
    # Nemotron-3 burns reasoning tokens before the answer; 16 -> empty content -> hash.
    assert seen["max_tokens"] >= 256


def test_build_responses_shapes_and_health_dependence():
    healthy = {
        "learning_rate": 2e-5,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
    }
    broken = {"learning_rate": 0.1, "batch_size": 8, "lora_r": 8, "lora_alpha": 16, "num_epochs": 3}
    base_h, ft_h = build_responses(healthy)
    base_b, ft_b = build_responses(broken)
    assert len(base_h) == len(DEFAULT_PROMPTS) == 5
    assert ft_h[DEFAULT_PROMPTS[0]] != ft_b[DEFAULT_PROMPTS[0]]
    assert "structured" in ft_h[DEFAULT_PROMPTS[0]].lower() or "1)" in ft_h[DEFAULT_PROMPTS[0]]
    assert "uncertain" in ft_b[DEFAULT_PROMPTS[0]]


def test_build_responses_answers_are_distinct_and_nonempty():
    healthy_cfg = {"learning_rate": 0.01}
    broken_cfg = {"learning_rate": 0.01, "lora_r": 32, "num_epochs": 10}
    base_h, ft_h = build_responses(healthy_cfg)
    _, ft_b = build_responses(broken_cfg)
    for p in DEFAULT_PROMPTS:
        assert len(base_h[p]) > 40 and len(ft_h[p]) > 40 and len(ft_b[p]) > 40
        assert len({base_h[p], ft_h[p], ft_b[p]}) == 3


def test_run_eval_suite_regression_flag():
    # base always scores 9, fine-tuned always scores 3 -> regression True
    llm = FakeLLM(["9"] * 5 + ["3"] * 5)
    base, ft = build_responses({"learning_rate": 2e-5, "batch_size": 8})
    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=llm)
    assert len(results["base_scores"]) == 5
    assert results["regression_flagged"] is True
    assert all(0.0 <= s <= 10.0 for s in results["base_scores"] + results["fine_tuned_scores"])
    assert all(m == "base:real / ft:real" for m in results["modes"])


def test_fraction_replies_parse_numerator():
    llm = FakeLLM(["Score: 8/10", "7 out of 10"])
    assert score_response("p", "r", llm=llm).score == 8.0
    assert score_response("p", "r", llm=llm).score == 7.0


def test_suite_caps_at_five_prompts():
    llm = FakeLLM(["5"] * 15)  # 10 consumed for 5 prompts x 2 calls; 6th would take 12
    extra = DEFAULT_PROMPTS + ["sixth prompt"]
    base = {p: "b" for p in extra}
    ft = {p: "f" for p in extra}
    results = run_eval_suite(extra, base, ft, llm=llm)
    assert len(results["prompts"]) == 5
    assert len(llm.replies) == 5  # only 5 prompts scored


def test_suite_without_llm_reports_hash_modes():
    base, ft = build_responses({"learning_rate": 2e-5, "batch_size": 8})
    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=None)
    assert all(m == "base:hash / ft:hash" for m in results["modes"])
    assert results["regression_flagged"] is False
