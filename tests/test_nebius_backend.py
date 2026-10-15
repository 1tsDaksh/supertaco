"""Nebius judge backend: env validation, orchestration, labeled failures."""

import json

import pytest

from supertaco.errors import JudgeBackendError
from supertaco.eval.harness import DEFAULT_PROMPTS, run_eval_suite
from supertaco.eval.nebius_backend import NebiusJudgeBackend
from supertaco.gpu.nebius_jobs import JobOutcome


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


def _batch_scores():
    prompts = [f"p{i}" for i in range(5)]
    return {
        "prompts": prompts,
        "base_scores": [8.0] * 5,
        "fine_tuned_scores": [9.0] * 5,
        "modes": ["base:real / ft:real"] * 5,
        "call_log": [
            {
                "prompt": p,
                "variant": v,
                "mode": "real",
                "tokens": 11,
                "latency": 0.2,
                "error": None,
                "response_preview": "9",
            }
            for p in prompts
            for v in ("base", "fine_tuned")
        ],
    }


class FakeJobs:
    def __init__(self, outcome=None):
        self.requests = []
        self.outcome = outcome or JobOutcome(True, "aijob-9", "COMPLETED")

    def run(self, req):
        self.requests.append(req)
        return self.outcome


class FakeArtifacts:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else _batch_scores()
        self.put_keys = []
        self.deleted = []

    def ensure_bucket(self):
        return "supertaco-judge-artifacts"

    def presign_put(self, key):
        self.put_keys.append(key)
        return f"https://s3.example/{key}"

    def download_json(self, key):
        return self.payload

    def delete_object(self, key):
        self.deleted.append(key)


class FakeSecrets:
    def __init__(self):
        self.calls = []

    def ensure_tf_secret(self, project_id, name, tf_key):
        self.calls.append((project_id, name))
        return ("secret-1", "ver-1")


def _backend(**kwargs):
    kwargs.setdefault("project_id", "project-1")
    kwargs.setdefault("tf_base_url", "https://tf.example/v1")
    kwargs.setdefault("tf_api_key", "tf-key")
    kwargs.setdefault("jobs", FakeJobs())
    kwargs.setdefault("artifacts", FakeArtifacts())
    kwargs.setdefault("secrets", FakeSecrets())
    return NebiusJudgeBackend(**kwargs)


def test_backend_judge_returns_harness_shape_and_job_backend_label():
    backend = _backend()
    prompts = [f"p{i}" for i in range(5)]
    base = {p: f"b{i}" for i, p in enumerate(prompts)}
    ft = {p: f"f{i}" for i, p in enumerate(prompts)}
    result = backend.judge(prompts, base, ft)
    assert result["judge_backend"] == "nebius_job"
    assert result["base_scores"] == [8.0] * 5
    assert result["fine_tuned_scores"] == [9.0] * 5
    assert result["prompts"] == prompts
    assert len(result["call_log"]) == 10
    assert result["call_log"][0]["judge_backend"] == "nebius_job"
    assert result["call_log"][0]["job_id"] == "aijob-9"
    assert result["call_log"][0]["model_key"] == "judge"


def test_backend_injects_script_responses_and_secret_env():
    jobs = FakeJobs()
    artifacts = FakeArtifacts()
    backend = _backend(jobs=jobs, artifacts=artifacts)
    prompts = [f"p{i}" for i in range(5)]
    base = {p: "b" for p in prompts}
    ft = {p: "f" for p in prompts}
    backend.judge(prompts, base, ft)
    req = jobs.requests[0]
    assert req.image == "python:3.12-slim"
    assert req.platform == "cpu-d3"
    assert req.preset == "2vcpu-8gb"
    assert req.command == "python"
    assert req.args == "/supertaco/judge_job.py"
    paths = [f["container_path"] for f in req.files]
    assert paths == ["/supertaco/judge_job.py", "/supertaco/responses.json"]
    injected = json.loads(
        next(f["content"] for f in req.files if f["container_path"].endswith("responses.json"))
    )
    assert injected["prompts"] == prompts
    assert injected["base"] == base
    assert injected["fine_tuned"] == ft
    env = {e["name"]: e for e in req.env}
    assert env["TOKEN_FACTORY_BASE_URL"]["value"] == "https://tf.example/v1"
    assert env["TOKEN_FACTORY_API_KEY"]["mysterybox"] == {
        "secret_id": "secret-1",
        "version_id": "ver-1",
    }
    assert env["SCORES_PUT_URL"]["value"].startswith("https://s3.example/")
    assert env["RESPONSES_PATH"]["value"] == "/supertaco/responses.json"


def test_backend_job_failure_raises_and_cleans_object():
    artifacts = FakeArtifacts()
    backend = _backend(
        jobs=FakeJobs(JobOutcome(False, "aijob-9", "FAILED", error="disk full")),
        artifacts=artifacts,
    )
    with pytest.raises(JudgeBackendError, match="disk full"):
        backend.judge(["p"], {"p": "b"}, {"p": "f"})
    assert artifacts.deleted


def test_backend_malformed_scores_raise():
    artifacts = FakeArtifacts(payload={"nope": 1})
    backend = _backend(artifacts=artifacts)
    with pytest.raises(JudgeBackendError, match="scores"):
        backend.judge(["p"], {"p": "b"}, {"p": "f"})
    assert artifacts.deleted


def test_backend_missing_env_raises_at_construction(monkeypatch):
    from supertaco.settings import settings

    monkeypatch.setattr(settings, "nebius_project_id", "")
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    with pytest.raises(JudgeBackendError, match="NEBIUS_PROJECT_ID"):
        NebiusJudgeBackend(
            tf_base_url="https://tf.example/v1",
            tf_api_key="k",
            jobs=FakeJobs(),
            artifacts=FakeArtifacts(),
            secrets=FakeSecrets(),
        )


def test_backend_deletes_job_artifact_after_success():
    artifacts = FakeArtifacts()
    backend = _backend(artifacts=artifacts)
    backend.judge(
        ["p0", "p1", "p2", "p3", "p4"],
        {f"p{i}": "b" for i in range(5)},
        {f"p{i}": "f" for i in range(5)},
    )
    assert len(artifacts.deleted) == 1


def test_backend_download_error_still_raises_judge_backend_error():
    class BoomArtifacts(FakeArtifacts):
        def download_json(self, key):
            raise RuntimeError("connection reset by peer")

    backend = _backend(artifacts=BoomArtifacts())
    with pytest.raises(JudgeBackendError, match="connection reset"):
        backend.judge(
            ["p0", "p1", "p2", "p3", "p4"],
            {f"p{i}": "b" for i in range(5)},
            {f"p{i}": "f" for i in range(5)},
        )


def test_backend_malformed_call_log_entries_do_not_crash():
    payload = _batch_scores()
    payload["call_log"] = ["not-a-dict", {"prompt": "p0", "response_preview": 123}]
    backend = _backend(artifacts=FakeArtifacts(payload=payload))
    result = backend.judge(
        [f"p{i}" for i in range(5)],
        {f"p{i}": "b" for i in range(5)},
        {f"p{i}": "f" for i in range(5)},
    )
    assert result["judge_backend"] == "nebius_job"


def test_logs_dir_defaults_to_repo_root(monkeypatch):
    monkeypatch.undo()  # drop the conftest seam patch to see the real default
    from supertaco.eval import nebius_backend as nb_module
    from supertaco.paths import repo_root

    assert nb_module._logs_dir() == repo_root() / "logs"
