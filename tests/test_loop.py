"""Real training loop: transport seam, events, budget, eval gate."""

import json

import pytest

from supertaco.errors import ConfigurationError
from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.loop import (
    AttemptResult,
    extract_loss_points,
    validate_config,
)

HEALTHY_LOG = (
    "config: {'model': 'Qwen/Qwen2.5-0.5B-Instruct'}\n"
    "gpu: Tesla T4\n"
    "step 4 loss 1.9218\n"
    "step 8 loss 1.6969\n"
    "step 12 loss 1.3245\n"
    "training finished\n"
)

NAN_LOG = "step 4 loss NaN\nstep 8 loss NaN\ntraining finished\n"


def _block(base: dict, ft: dict) -> str:
    return f"{RESPONSES_START}\n{json.dumps({'base': base, 'fine_tuned': ft})}\n{RESPONSES_END}"


def _answers(score_word: str) -> dict:
    return {p: f"{score_word} answer" for p in ["a", "b", "c", "d", "e"]}


def healthy_responses() -> str:
    # prompts are substituted by the loop; build keys dynamically in transport fakes
    return ""


class FakeTransport:
    """Returns scripted AttemptResults in order; last entry repeats."""

    def __init__(self, results, fail_on=None):
        self.results = list(results)
        self.fail_on = fail_on or []
        self.calls = []

    def run_attempt(self, config, prompts, on_line=None):
        self.calls.append(dict(config))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("transport exploded")
        idx = min(len(self.calls) - 1, len(self.results) - 1)
        res = self.results[idx]
        if callable(res):
            return res(prompts)
        return res


def make_result(logs, responses_payload=None, artifact=None, error=None):
    text = logs
    if responses_payload is not None:
        text += "\n" + _block(responses_payload["base"], responses_payload["fine_tuned"])
    return AttemptResult(logs=text, artifact=artifact, error=error)


def responses_for(prompts, base="base", ft="fine"):
    return {
        "base": {p: f"{base} answer to {p}" for p in prompts},
        "fine_tuned": {p: f"{ft} answer to {p}" for p in prompts},
    }


class FakeLLM:
    """Judge scores alternate base/ft calls; classify/patch are deterministic."""

    def __init__(self, healthy=True):
        self.call_log = []
        self.healthy = healthy
        self._n = 0

    def classify_failure(self, logs):
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        is_base = self._n % 2 == 0
        self._n += 1
        if self.healthy:
            return "9/10" if not is_base else "5/10"
        return "3/10" if not is_base else "9/10"  # ft worse -> regression


def collect_events():
    events = []
    return events, lambda ev: events.append(ev)


# ---------- Task 2: helpers ----------


def test_validate_config_rejects_missing_learning_rate():
    with pytest.raises(ConfigurationError, match="learning_rate"):
        validate_config({"batch_size": 8})


def test_validate_config_rejects_empty():
    with pytest.raises(ConfigurationError):
        validate_config({})


def test_extract_loss_points_from_real_lines():
    pts = extract_loss_points("step 4 loss 1.9218\nstep 8 loss 0.5")
    assert pts == [1.9218, 0.5]


def test_extract_loss_points_preserves_nan():
    pts = extract_loss_points("step 4 loss NaN")
    assert len(pts) == 1 and pts[0] != pts[0]  # NaN
