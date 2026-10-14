"""Nebius judge backend: env validation, orchestration, labeled failures."""

from supertaco.errors import JudgeBackendError
from supertaco.eval.harness import DEFAULT_PROMPTS, run_eval_suite


def test_judge_backend_error_is_exception_with_message():
    err = JudgeBackendError("missing NEBIUS_PROJECT_ID; set it in .env")
    assert isinstance(err, Exception)
    assert "NEBIUS_PROJECT_ID" in str(err)


def _resp_maps():
    base = {p: f"base answer to {p}" for p in DEFAULT_PROMPTS}
    ft = {p: f"ft answer to {p}" for p in DEFAULT_PROMPTS}
    return base, ft


def test_run_eval_suite_with_judge_callable_uses_batch_result():
    base, ft = _resp_maps()
    seen = {}

    def fake_judge(prompts, base_responses, ft_responses):
        seen["prompts"] = list(prompts)
        return {
            "prompts": list(prompts),
            "base_scores": [8.0] * 5,
            "fine_tuned_scores": [9.0] * 5,
            "modes": ["base:real / ft:real"] * 5,
            "call_log": [{"model_key": "judge"}],
            "judge_backend": "nebius_job",
        }

    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=None, judge=fake_judge)
    assert seen["prompts"] == DEFAULT_PROMPTS[:5]
    assert results["judge_backend"] == "nebius_job"
    assert results["fine_tuned_scores"] == [9.0] * 5
    assert results["regression_flagged"] is False
    assert results["call_log"] == [{"model_key": "judge"}]


def test_run_eval_suite_without_judge_stays_direct():
    base, ft = _resp_maps()

    class LLM:
        call_log = []

        def _call(self, *a, **kw):
            return "8"

    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, llm=LLM())
    assert results["judge_backend"] == "direct"
    assert results["call_log"] == []


def test_run_eval_suite_judge_regression_flag():
    base, ft = _resp_maps()

    def reg_judge(prompts, base_responses, ft_responses):
        return {
            "prompts": list(prompts),
            "base_scores": [9.0] * 5,
            "fine_tuned_scores": [3.0] * 5,
            "modes": ["base:real / ft:real"] * 5,
            "call_log": [],
            "judge_backend": "nebius_job",
        }

    results = run_eval_suite(DEFAULT_PROMPTS, base, ft, judge=reg_judge)
    assert results["regression_flagged"] is True
