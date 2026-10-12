"""Real judge scoring with hash fallback, clamping, and suite regression flag."""

from supertaco.eval.harness import DEFAULT_PROMPTS, run_eval_suite
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
    llm = FakeLLM(["no numbers here", "still no numbers"])
    result = score_response("p", "r", llm=llm)
    assert result.mode == "hash"
    assert 2.0 <= result.score <= 10.0
    assert len(llm.call_log) == 2  # one retry, then hash


class FallbackLLM:
    """HTTP-level failure: every call logged as fallback, replies unparseable."""

    def __init__(self):
        self.call_log = []

    def _call(self, model_key, prompt, max_tokens=1024, temperature=0.0):
        self.call_log.append(
            {
                "model_key": model_key,
                "mode": "fallback",
                "tokens": 0,
                "latency": 0.0,
                "error": "ConnectError: connection refused",
            }
        )
        return '{"fix": "lower_learning_rate", "reason": "model suggests LR reduction"}'


def test_transport_failure_after_retry_reports_fallback_mode():
    llm = FallbackLLM()
    result = score_response("p", "r", llm=llm)
    assert result.mode == "fallback"
    assert 2.0 <= result.score <= 10.0
    assert len(llm.call_log) == 2


def test_fallback_mode_with_no_error_reports_unknown_transport_error():
    class SilentFallbackLLM(FallbackLLM):
        def _call(self, model_key, prompt, max_tokens=1024, temperature=0.0):
            self.call_log.append(
                {
                    "model_key": model_key,
                    "mode": "fallback",
                    "tokens": 0,
                    "latency": 0.0,
                    "error": None,
                }
            )
            return "no numbers here"

    result = score_response("p", "r", llm=SilentFallbackLLM())
    assert result.mode == "fallback"
    assert "unknown transport error" in result.error
    assert "None" not in result.error


def test_final_real_unparseable_after_earlier_http_failure_reports_hash():
    class MixedLLM:
        def __init__(self):
            self.call_log = []
            self.calls = 0

        def _call(self, model_key, prompt, max_tokens=1024, temperature=0.0):
            self.calls += 1
            mode = "fallback" if self.calls == 1 else "real"
            error = "ConnectError: connection refused" if mode == "fallback" else None
            self.call_log.append(
                {"model_key": model_key, "mode": mode, "tokens": 0, "latency": 0.0, "error": error}
            )
            return "" if mode == "fallback" else "no numbers here"

    result = score_response("p", "r", llm=MixedLLM())
    assert result.mode == "hash"


def test_suite_reports_fallback_modes_for_transport_failures():
    base = {p: f"base answer to {p}" for p in DEFAULT_PROMPTS}
    ft = {p: f"ft answer to {p}" for p in DEFAULT_PROMPTS}
    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=FallbackLLM())
    assert all(m == "base:fallback / ft:fallback" for m in results["modes"])


def test_empty_reply_retries_once_then_scores():
    llm = FakeLLM(["", "7"])
    result = score_response("p", "r", llm=llm)
    assert result.score == 7.0
    assert result.mode == "real"
    assert len(llm.call_log) == 2


def test_garbage_reply_retries_once_then_scores():
    llm = FakeLLM(["no numbers here", "8/10"])
    result = score_response("p", "r", llm=llm)
    assert result.score == 8.0
    assert result.mode == "real"
    assert len(llm.call_log) == 2


def test_valid_reply_does_not_retry():
    llm = FakeLLM(["8"])
    result = score_response("p", "r", llm=llm)
    assert result.score == 8.0
    assert len(llm.call_log) == 1


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


def test_run_eval_suite_regression_flag():
    # base always scores 9, fine-tuned always scores 3 -> regression True
    llm = FakeLLM(["9"] * 5 + ["3"] * 5)
    base = {p: f"base answer to {p}" for p in DEFAULT_PROMPTS}
    ft = {p: f"ft answer to {p}" for p in DEFAULT_PROMPTS}
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
    base = {p: f"base answer to {p}" for p in DEFAULT_PROMPTS}
    ft = {p: f"ft answer to {p}" for p in DEFAULT_PROMPTS}
    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=None)
    assert all(m == "base:hash / ft:hash" for m in results["modes"])
    assert results["regression_flagged"] is False
