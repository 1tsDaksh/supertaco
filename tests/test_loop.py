"""Real training loop: transport seam, events, budget, eval gate."""

import json
import tempfile

import pytest

from supertaco.agent.citations import Citation, CitationClient
from supertaco.errors import ConfigurationError, JudgeBackendError
from supertaco.eval.harness import DEFAULT_PROMPTS, RESPONSES_END, RESPONSES_START
from supertaco.loop import (
    AttemptResult,
    _resolve_runs_dir,
    extract_loss_points,
    run_training_loop,
    validate_config,
)
from supertaco.paths import repo_root


def _tmp_runs():
    d = tempfile.mkdtemp(prefix="st_runs_")
    return d


def test_resolve_runs_dir_anchors_relative_to_repo_root():
    assert _resolve_runs_dir("configs/runs") == repo_root() / "configs" / "runs"


def test_resolve_runs_dir_passes_absolute_through(tmp_path):
    assert _resolve_runs_dir(tmp_path) == tmp_path
    assert _resolve_runs_dir(str(tmp_path)) == tmp_path


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

    def __init__(self, healthy=True, heal_after_first_eval=False):
        self.call_log = []
        self.healthy = healthy
        self.heal_after_first_eval = heal_after_first_eval
        self._n = 0

    def classify_failure(self, logs):
        self.call_log.append(
            {"model_key": "classify", "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        self.call_log.append(
            {"model_key": "patch", "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        return {"fix": f"fix-{failure_key}", "reason": "because"}

    def _call(self, model_key, prompt, **kw):
        self.call_log.append(
            {"model_key": model_key, "mode": "real", "tokens": 1, "latency": 0.01, "error": None}
        )
        is_base = self._n % 2 == 0
        self._n += 1
        # heal flag takes precedence: healthy arg ignored when True
        if self.heal_after_first_eval:
            healthy_now = self._n > 10  # first eval = 10 calls (5 prompts x base/ft) regresses
        else:
            healthy_now = self.healthy
        if healthy_now:
            return "9/10" if not is_base else "5/10"
        return "3/10" if not is_base else "9/10"  # ft worse -> regression


def collect_events():
    events = []
    return events, lambda ev: events.append(ev)


def _cite_raw():
    return [
        {"title": "T1", "url": "https://a.example", "content": "one"},
        {"title": "T2", "url": "https://b.example", "content": "two"},
    ]


class FakeCitationClient:
    """Records every query; returns two fixed citations."""

    def __init__(self):
        self.queries = []

    def search_citations(self, query, *, max_results=2):
        self.queries.append(query)
        return [
            Citation(title="T1", url="https://a.example"),
            Citation(title="T2", url="https://b.example"),
        ]


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


def test_extract_loss_points_preserves_lowercase_nan():
    pts = extract_loss_points("step 4 loss nan\nstep 8 loss nan")
    assert len(pts) == 2
    assert all(p != p for p in pts)  # NaN checks


# ---------- Task 3: run_training_loop ----------


def test_happy_path_first_attempt_success():
    transport = FakeTransport(
        [
            lambda prompts: make_result(
                HEALTHY_LOG, responses_for(prompts), artifact="runs/run_001/artifacts/eval.zip"
            )
        ]
    )
    llm = FakeLLM()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is True
    assert result.attempts == 1
    assert result.failure_key is None
    assert result.eval_results is not None
    assert result.eval_results["regression_flagged"] is False
    assert result.artifact is not None
    assert len(result.attempt_ledger) == 1
    assert result.attempt_ledger[0].failure_key is None
    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert "job_launched" in types and "logs_produced" in types
    assert types[-1] == "run_succeeded"
    launch = next(e for e in events if e.type == "job_launched")
    assert launch.data["payload"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert launch.data["payload"]["config"]["learning_rate"] == 2e-4


def test_transport_error_becomes_run_failed_never_raises():
    transport = FakeTransport([AttemptResult(logs="", error="Colab CLI is not authenticated")])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "not authenticated" in (result.error or "")
    assert [e.type for e in events][-1] == "run_failed"


# ---------- Task 3 review fixes ----------


def test_transport_exception_fails_fast_with_ledger():
    transport = FakeTransport([make_result(HEALTHY_LOG)], fail_on=[1])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "transport exploded" in (result.error or "")
    assert result.attempts == 1
    assert [e.type for e in events][-1] == "run_failed"
    assert len(result.attempt_ledger) == 1  # evidence survives fail-fast (fix 3)


def test_non_dict_config_never_raises():
    transport = FakeTransport([make_result(HEALTHY_LOG)])
    result = run_training_loop(
        "not-a-dict",
        transport=transport,
        llm=FakeLLM(),
        on_event=lambda ev: None,
    )
    assert result.success is False
    assert "non-empty mapping" in (result.error or "")
    assert transport.calls == []


def test_negative_max_retries_returns_failed_result():
    transport = FakeTransport([make_result(HEALTHY_LOG)])
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        max_retries=-1,
    )
    assert result.success is False
    assert "MaxRetriesExceeded" in (result.error or "")
    assert result.attempts == 0


# ---------- Task 4: plan pins ----------


def test_nan_failure_patches_and_heals():
    transport = FakeTransport(
        [
            make_result(NAN_LOG),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is True
    assert result.attempts == 2
    types = [e.type for e in events]
    assert (
        types.index("failure_detected")
        < types.index("patch_written")
        < types.index("retry_scheduled")
    )
    assert types.count("job_launched") == 2
    assert result.final_config["learning_rate"] == 0.01  # playbook NAN_LOSS: lr / 10
    assert len(result.configs_written) == 1
    rec = result.attempt_ledger[0]
    assert rec.failure_key == "NAN_LOSS"
    assert rec.verdict == "NAN_LOSS"
    assert rec.config_after["learning_rate"] == 0.01
    # classify-mode pin (review addendum 7 — needs the FakeLLM change):
    classified = next(e for e in events if e.type == "classified")
    assert classified.data["mode"] == "real"
    assert classified.data["diverged"] is False
    assert transport.calls[0]["learning_rate"] == 0.1  # first attempt ran the broken config
    assert transport.calls[1]["learning_rate"] == 0.01  # relaunch received the PATCHED config
    assert "patch_proposed" in types  # frozen 10-event catalog includes this
    assert len(result.attempt_ledger[0].loss_points) == 2  # NAN_LOG has 2 loss lines; D5 evidence


def test_budget_exhausted_returns_run_failed_with_ledger():
    transport = FakeTransport([make_result(NAN_LOG)])  # always NaN
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
        max_retries=2,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is False
    assert "MaxRetriesExceeded" in (result.error or "")
    assert result.attempts == 3  # initial + 2 relaunches
    assert len(result.attempt_ledger) == 3
    assert all(r.failure_key == "NAN_LOSS" for r in result.attempt_ledger)
    assert [e.type for e in events][-1] == "run_failed"
    # patched configs feed RELAUNCHES 2 and 3; attempt 3 breaks before patching
    assert len(result.configs_written) == 2


# ---------- Task 4: review-required tests ----------


def test_responses_missing_prompt_key_honest_failure():
    prompts_only_first = {
        "base": {DEFAULT_PROMPTS[0]: "base answer"},
        "fine_tuned": {DEFAULT_PROMPTS[0]: "ft answer"},
    }
    transport = FakeTransport([make_result(HEALTHY_LOG, prompts_only_first)])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "missing prompt key" in (result.error or "")
    assert len(result.attempt_ledger) == 1
    assert [e.type for e in events][-1] == "run_failed"


def test_failed_result_carries_prior_eval_results():
    transport = FakeTransport(
        [
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
            make_result(HEALTHY_LOG),
        ],
        fail_on=[2],
    )
    result = run_training_loop(
        {
            "learning_rate": 2e-4,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 3,
            "model": "qwen2.5-0.5b",
        },
        transport=transport,
        llm=FakeLLM(healthy=False),
        on_event=lambda ev: None,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is False
    assert "transport exploded" in (result.error or "")
    assert len(result.attempt_ledger) == 2
    assert result.attempt_ledger[0].failure_key == "EVAL_REGRESSION"
    assert result.attempt_ledger[1].failure_key is None
    assert result.eval_results is not None
    assert result.eval_results["regression_flagged"] is True


def test_error_result_captures_partial_log_points():
    transport = FakeTransport(
        [AttemptResult(logs="step 4 loss 1.9\ndownload failed", error="download failed")]
    )
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=lambda ev: None,
    )
    assert result.success is False
    assert len(result.attempt_ledger) == 1
    assert len(result.attempt_ledger[0].loss_points) == 1
    assert result.eval_results is None


def test_error_result_with_none_logs_never_raises():
    transport = FakeTransport([AttemptResult(logs=None, error="download failed")])
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=lambda ev: None,
    )
    assert result.success is False
    assert "download failed" in (result.error or "")
    assert len(result.attempt_ledger) == 1


def test_run_started_emitted_before_llm_construction(monkeypatch):
    events, on_event = collect_events()
    seen_at_construction: list[str] = []

    class RecordingLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            seen_at_construction.extend(e.type for e in events)

    monkeypatch.setattr("supertaco.loop.make_llm", RecordingLLM)
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=FakeTransport([make_result(HEALTHY_LOG)]),  # no responses block -> honest failure
        on_event=on_event,
    )
    assert result.success is False
    assert seen_at_construction, "make_llm must be constructed (proves llm=None path)"
    assert seen_at_construction[0] == "run_started"


# ---------- Task 5: eval gate pins ----------


def test_eval_regression_triggers_patch_then_heals():
    transport = FakeTransport(
        [
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM(heal_after_first_eval=True)  # eval 1 regressed, eval 2 healthy
    events, on_event = collect_events()
    result = run_training_loop(
        {
            "learning_rate": 2e-4,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 3,
            "model": "qwen2.5-0.5b",
        },
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    failure_events = [e for e in events if e.type == "failure_detected"]
    assert failure_events[0].data["failure_key"] == "EVAL_REGRESSION"
    patch = next(e for e in events if e.type == "patch_written")
    assert patch.data["failure_key"] == "EVAL_REGRESSION"
    assert patch.data["after"]["lora_r"] == 4  # max(4, 8//2) floors at 4
    assert patch.data["after"]["lora_alpha"] == 8  # max(8, 16//2) floors at 8
    assert patch.data["after"]["num_epochs"] == 2  # -1 epoch
    assert patch.data["after"]["learning_rate"] == 1e-4  # 2e-4 / 2
    assert patch.data["after"]["lora_dropout"] == 0.1  # default 0.05 doubled
    assert result.success is True
    assert result.attempts == 2
    # review addendum: classify sees HEALTHY logs on an eval regression -> verdict must diverge
    classified = next(e for e in events if e.type == "classified")
    assert classified.data["failure_key"] == "EVAL_REGRESSION"
    assert classified.data["diverged"] is True


def test_healthy_but_no_responses_block_is_honest_failure():
    transport = FakeTransport([make_result(HEALTHY_LOG)])  # no block
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    assert "RESPONSES_JSON" in (result.error or "")
    assert [e.type for e in events][-1] == "run_failed"
    assert len(result.attempt_ledger) == 1  # partial record survives fail-fast (D5)
    assert len(result.attempt_ledger[0].loss_points) == 3  # HEALTHY_LOG has 3 loss lines


def test_log_failure_and_eval_regression_share_one_budget():
    transport = FakeTransport(
        [
            make_result(NAN_LOG),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM(healthy=False)  # ALWAYS regresses: eval 2 can never heal within max_retries=1
    events, on_event = collect_events()
    result = run_training_loop(
        {
            "learning_rate": 0.1,
            "batch_size": 8,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": 3,
            "model": "qwen2.5-0.5b",
        },
        transport=transport,
        llm=llm,
        on_event=on_event,
        max_retries=1,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is False
    assert "MaxRetriesExceeded" in (result.error or "")
    assert result.attempts == 2  # 1 log failure + 1 eval regression SHARE the single relaunch
    assert [r.failure_key for r in result.attempt_ledger] == ["NAN_LOSS", "EVAL_REGRESSION"]
    assert len(result.configs_written) == 1  # attempt 2 breaks before patching
    assert [e.type for e in events][-1] == "run_failed"
    assert events[-1].data["attempts"] == 2  # run_failed payload: attempts key
    assert events[-1].data["path"]  # path key exists (dashboard reads it with .get)


# ---------- Task 6: pre-launch model validation pins (spec D9) ----------


def test_unknown_model_honest_runtime_failure():
    transport = FakeTransport([lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts))])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "mistral-7b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    # resolve_model raises ConfigurationError("model 'mistral-7b' is not T4-viable; ...")
    assert "T4-viable" in (result.error or "")
    assert transport.calls == []  # validation fired BEFORE any launch (D9)
    assert result.configs_written == []  # nothing written on pre-launch failure
    assert [e.type for e in events][-1] == "run_failed"
    assert "job_launched" not in [e.type for e in events]


def test_missing_model_key_is_config_error():
    transport = FakeTransport([lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts))])
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4},  # no model key -> resolve_model(None)
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
    )
    assert result.success is False
    # resolve_model(None) raises "model None is not T4-viable; pick from: ..."
    assert "model None" in (result.error or "")
    assert "T4-viable" in (result.error or "")
    assert transport.calls == []


def test_custom_prompts_flow_to_transport_default_unchanged():
    seen: list[list[str]] = []

    class T:
        def run_attempt(self, config, prompts, on_line=None):
            seen.append(list(prompts))
            return AttemptResult(logs=HEALTHY_LOG)

    cfg = {"model": "qwen2.5-0.5b", "learning_rate": 2e-4}
    res = run_training_loop(
        cfg,
        transport=T(),
        llm=FakeLLM(),
        max_retries=0,
        runs_dir=_tmp_runs(),
        prompts=["goal prompt one"],
    )
    assert seen == [["goal prompt one"]]
    assert res.success is False  # no RESPONSES_JSON block: honest failure

    run_training_loop(cfg, transport=T(), llm=FakeLLM(), max_retries=0, runs_dir=_tmp_runs())
    assert seen[1] == DEFAULT_PROMPTS  # default path unchanged


def test_custom_prompts_reach_the_judge():
    transport = FakeTransport([lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts))])
    res = run_training_loop(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4},
        transport=transport,
        llm=FakeLLM(),
        max_retries=0,
        runs_dir=_tmp_runs(),
        prompts=["goal prompt one"],
    )
    assert res.success is True
    assert res.eval_results["prompts"] == ["goal prompt one"]


# ---------- hard VM crashes (exec exits 0) must reach the heal path ----------


def test_oom_crash_retries_with_halved_batch():
    oom_tail = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.06 GiB."
    transport = FakeTransport(
        [
            AttemptResult(
                logs=oom_tail,
                error=(
                    f"Download failed: no /content/lora_adapter.zip (training tail: {oom_tail})"
                ),
            ),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is True
    assert result.attempts == 2
    assert len(transport.calls) == 2
    second = transport.calls[1]
    assert second["batch_size"] == 4  # playbook OOM fix halves batch_size
    assert second["gradient_checkpointing"] is True
    assert any(c["model_key"] == "classify" for c in llm.call_log)
    assert any(c["model_key"] == "patch" for c in llm.call_log)
    types = [e.type for e in events]
    assert "failure_detected" in types and "patch_written" in types
    assert types.count("logs_produced") == 1  # the crash attempt emits no fake logs
    rec = result.attempt_ledger[0]
    assert rec.failure_key == "OOM"
    assert rec.config_after["batch_size"] == 4
    assert rec.config_after["gradient_checkpointing"] is True


def test_unrecognized_error_still_fails_fast():
    transport = FakeTransport(
        [AttemptResult(logs="zorp quuxle", error="zorp quuxle: totally unrecognized")]
    )
    llm = FakeLLM()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is False
    assert result.attempts == 1
    assert len(transport.calls) == 1  # D9: no retry for unrecognized failures
    assert result.failure_key is None
    assert llm.call_log == []  # no classify/patch on a fail-fast
    assert [e.type for e in events][-1] == "run_failed"


def test_cp1252_script_read_keeps_judge_prompts_in_sync(tmp_path):
    """Regression: the colab CLI reads the rendered script with the Windows
    locale codec (cp1252). A U+2011 hyphen from planner prompts used to
    mojibake on the VM, so the judge KeyError'd on the clean prompt key."""
    import ast
    import re

    from supertaco.gpu.transport import render_script

    template = repo_root() / "colab" / "train_lora.py"

    def cp1252_vm(prompts_arg):
        config = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4}
        script = render_script(config, list(prompts_arg), template, tmp_path / "renders")
        raw = script.read_bytes().decode("cp1252")
        match = re.search(r"EVAL_PROMPTS = (\[.*?\])\r?\n", raw, re.S)
        vm_prompts = ast.literal_eval(match.group(1))
        return make_result(HEALTHY_LOG, responses_for(vm_prompts))

    prompts = [
        "Explain the elements of negligence in a personal injury case.",
        (
            "Draft a short clause for a non\u2011disclosure agreement "
            "that protects confidential information."
        ),
        "Summarize Miranda v. Arizona.",
    ]
    transport = FakeTransport([cp1252_vm])
    llm = FakeLLM()
    result = run_training_loop(
        {"learning_rate": 2e-4, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        runs_dir=str(tmp_path / "runs"),
        prompts=prompts,
    )
    assert result.success, result.error


# ---------- Task 2: citations wired into the heal path ----------


def test_citations_searched_attached_to_patch_event_and_ledger():
    transport = FakeTransport(
        [
            make_result(NAN_LOG),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    llm = FakeLLM()
    cites = CitationClient(
        "tvly-test", search=lambda q, max_results=2: _cite_raw(), call_log=llm.call_log
    )
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
        citations=cites,
    )
    assert result.success is True
    # query shape from WIN_PLAN §5: "<KEY>: <playbook description> pytorch LoRA fine-tuning"
    cit_ev = next(e for e in events if e.type == "citations_found")
    assert cit_ev.data["failure_key"] == "NAN_LOSS"
    assert cit_ev.data["query"].startswith("NAN_LOSS:")
    assert "pytorch LoRA fine-tuning" in cit_ev.data["query"]
    assert "Loss is NaN" in cit_ev.data["query"]  # PLAYBOOK NAN_LOSS description present
    patch_ev = next(e for e in events if e.type == "patch_proposed")
    assert [c["title"] for c in patch_ev.data["citations"]] == ["T1", "T2"]
    assert result.attempt_ledger[0].citations[0]["url"] == "https://a.example"
    assert result.attempt_ledger[1].citations == []  # healthy attempt: no search
    # audit entries flow into the LoopResult
    assert any(c.get("model_key") == "tavily" for c in result.llm_calls)


def test_citations_searched_once_per_failure_key_and_reused_on_retry():
    transport = FakeTransport([make_result(NAN_LOG)])  # always NaN -> 3 attempts
    cites = FakeCitationClient()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
        max_retries=2,
        citations=cites,
    )
    assert result.success is False  # budget exhausted
    assert len(cites.queries) == 1  # 3 failed attempts, ONE search
    assert len([e for e in events if e.type == "citations_found"]) == 1
    assert all(r.citations for r in result.attempt_ledger)  # cache reused for attempts 2-3
    patches = [e for e in events if e.type == "patch_proposed"]
    assert patches and all(e.data["citations"] for e in patches)


def test_no_citations_client_means_no_search_behavior_change():
    # conftest stubs client_from_settings -> None, so the default path skips entirely
    transport = FakeTransport(
        [
            make_result(NAN_LOG),
            lambda prompts: make_result(HEALTHY_LOG, responses_for(prompts)),
        ]
    )
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
    )
    assert result.success is True
    assert "citations_found" not in [e.type for e in events]
    assert result.attempt_ledger[0].citations == []
    patch_ev = next(e for e in events if e.type == "patch_proposed")
    assert patch_ev.data["citations"] == []


def test_citations_searched_again_when_failure_key_changes():
    oom_log = (
        "Traceback (most recent call last):\n"
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    )
    transport = FakeTransport([make_result(oom_log), make_result(NAN_LOG)])
    cites = FakeCitationClient()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 0.1, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
        citations=cites,
    )
    assert result.success is False  # budget exhausted after mixed failures
    assert len(cites.queries) == 2  # OOM then NAN_LOSS -> two searches
    assert cites.queries[0].startswith("OOM:")
    assert cites.queries[1].startswith("NAN_LOSS:")
    cit_evs = [e for e in events if e.type == "citations_found"]
    assert [e.data["failure_key"] for e in cit_evs] == ["OOM", "NAN_LOSS"]
    assert result.attempt_ledger[0].failure_key == "OOM"
    assert result.attempt_ledger[1].failure_key == "NAN_LOSS"
    # each record carries citations (both keys searched; no cross-contamination of queries)
    assert result.attempt_ledger[0].citations
    assert result.attempt_ledger[1].citations


def test_playbook_miss_crash_searches_citations_and_stays_honest():
    transport = FakeTransport(
        [AttemptResult(logs="zorp quuxle", error="zorp quuxle: totally unrecognized")]
    )
    llm = FakeLLM()
    cites = FakeCitationClient()
    events, on_event = collect_events()
    result = run_training_loop(
        {"learning_rate": 2e-4, "batch_size": 8, "model": "qwen2.5-0.5b"},
        transport=transport,
        llm=llm,
        on_event=on_event,
        runs_dir=str(_tmp_runs()),
        citations=cites,
    )
    # honest failure is unchanged: fail fast, no classify/patch, no retry
    assert result.success is False
    assert result.failure_key is None
    assert len(transport.calls) == 1
    assert llm.call_log == []
    # but the human gets external context for the unknown crash
    assert len(cites.queries) == 1
    assert cites.queries[0].startswith("pytorch fine-tuning crash: ")
    assert "zorp quuxle" in cites.queries[0]
    cit_ev = next(e for e in events if e.type == "citations_found")
    assert cit_ev.data["failure_key"] is None
    assert cit_ev.data["citations"][0]["title"] == "T1"
    assert result.attempt_ledger[0].citations[0]["title"] == "T1"
    assert [e.type for e in events][-1] == "run_failed"


# ---------- Task 7: loop judge_backend seam + labeled direct fallback ----------


def _healthy_transport():
    return FakeTransport([make_result(HEALTHY_LOG, responses_for(DEFAULT_PROMPTS))])


def test_loop_judge_backend_direct_is_default(monkeypatch):
    # existing tests already cover this; explicit pin for the WS1.2 seam
    events, on_event = collect_events()
    result = run_training_loop(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4},
        transport=_healthy_transport(),
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=_tmp_runs(),
    )
    assert result.success
    assert result.eval_results["judge_backend"] == "direct"


def test_loop_nebius_backend_success_labels_job(monkeypatch):
    from supertaco.eval import nebius_backend as nb

    def fake_judge_backend_factory():
        class B:
            def judge(self, prompts, base, ft):
                return {
                    "prompts": list(prompts),
                    "base_scores": [5.0] * 5,
                    "fine_tuned_scores": [9.0] * 5,
                    "modes": ["base:real / ft:real"] * 5,
                    "call_log": [
                        {
                            "model_key": "judge",
                            "model_id": "nvidia/nemotron-3-super-120b-a12b",
                            "mode": "real",
                            "tokens": 1,
                            "latency": 0.1,
                            "error": None,
                            "timestamp": 0.0,
                            "response_preview": "9",
                            "judge_backend": "nebius_job",
                        }
                    ],
                    "judge_backend": "nebius_job",
                }

        return B()

    monkeypatch.setattr(nb, "NebiusJudgeBackend", lambda **kw: fake_judge_backend_factory())
    events, on_event = collect_events()
    result = run_training_loop(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4},
        transport=_healthy_transport(),
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=_tmp_runs(),
        judge_backend="nebius_job",
    )
    assert result.success
    assert result.eval_results["judge_backend"] == "nebius_job"
    # job call_log merged into the run's llm_calls
    assert any(c.get("judge_backend") == "nebius_job" for c in result.llm_calls)
    assert not any(ev.type == "judge_fallback" for ev in events)


def test_loop_nebius_backend_error_falls_back_labeled(monkeypatch):
    from supertaco.eval import nebius_backend as nb

    def boom(**kw):
        raise JudgeBackendError("job aijob-1 ended FAILED: disk full")

    monkeypatch.setattr(nb, "NebiusJudgeBackend", boom)
    events, on_event = collect_events()
    result = run_training_loop(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4},
        transport=_healthy_transport(),
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=_tmp_runs(),
        judge_backend="nebius_job",
    )
    assert result.success  # run still succeeds via direct judging
    assert result.eval_results["judge_backend"] == "direct_fallback"
    fallbacks = [ev for ev in events if ev.type == "judge_fallback"]
    assert len(fallbacks) == 1
    assert "disk full" in fallbacks[0].data["reason"]
    assert fallbacks[0].data["stage"] == "setup"


def test_loop_nebius_mid_judge_error_falls_back(monkeypatch):
    from supertaco.eval import nebius_backend as nb

    class HalfBroken:
        def judge(self, prompts, base, ft):
            raise JudgeBackendError("scores PUT failed: timeout")

    monkeypatch.setattr(nb, "NebiusJudgeBackend", lambda **kw: HalfBroken())
    events, on_event = collect_events()
    result = run_training_loop(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4},
        transport=_healthy_transport(),
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=_tmp_runs(),
        judge_backend="nebius_job",
    )
    assert result.success
    assert result.eval_results["judge_backend"] == "direct_fallback"
    assert any(
        "PUT failed" in ev.data.get("reason", "") for ev in events if ev.type == "judge_fallback"
    )
    fallback = next(ev for ev in events if ev.type == "judge_fallback")
    assert fallback.data["stage"] == "job"


def test_loop_nebius_missing_prompt_key_fails_gracefully(monkeypatch):
    from supertaco.eval import nebius_backend as nb

    class MissingKeyJudgeBackend:
        # real NebiusJudgeBackend never raises raw exceptions: a missing prompt
        # key surfaces as JudgeBackendError (its never-raise contract), which
        # triggers the loop's direct re-run — that re-run must still hit the
        # loop's KeyError branch and fail gracefully, not crash.
        def judge(self, prompts, base, ft):
            raise JudgeBackendError(
                "responses payload missing prompt key: "
                "'Give one reason eval scores can regress after training.'"
            )

    monkeypatch.setattr(nb, "NebiusJudgeBackend", lambda **kw: MissingKeyJudgeBackend())
    events, on_event = collect_events()
    # transport returns a responses payload missing one prompt key
    incomplete = responses_for(DEFAULT_PROMPTS[:-1])  # drop the last prompt's answers
    result = run_training_loop(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4},
        transport=FakeTransport([make_result(HEALTHY_LOG, incomplete)]),
        llm=FakeLLM(),
        on_event=on_event,
        runs_dir=_tmp_runs(),
        judge_backend="nebius_job",
    )
    assert result.success is False
    assert "missing prompt key" in result.error
