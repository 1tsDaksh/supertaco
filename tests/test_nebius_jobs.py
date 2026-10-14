"""Nebius job lifecycle: spec assembly, polling, timeout, always-delete."""

import pytest

from supertaco.gpu.nebius_jobs import (
    TERMINAL_STATE_NAMES,
    JobOutcome,
    JobRequest,
    NebiusJobsClient,
    _state_name,
    build_job_spec,
)


def _request(**overrides):
    kwargs = dict(
        name="supertaco-judge-1",
        image="python:3.12-slim",
        platform="cpu-d3",
        preset="2vcpu-8gb",
        disk_gib=10,
        command="python",
        args="/supertaco/judge_job.py",
        env=[
            {"name": "TOKEN_FACTORY_BASE_URL", "value": "https://tf.example/v1"},
            {
                "name": "TOKEN_FACTORY_API_KEY",
                "mysterybox": {"secret_id": "secret-1", "version_id": "ver-1"},
            },
        ],
        files=[
            {"container_path": "/supertaco/judge_job.py", "content": b"print(1)\n"},
            {"container_path": "/supertaco/responses.json", "content": b"{}"},
        ],
    )
    kwargs.update(overrides)
    return JobRequest(**kwargs)


def test_build_job_spec_assembles_injections_and_env():
    spec = build_job_spec(_request())
    assert spec.image == "python:3.12-slim"
    assert spec.platform == "cpu-d3"
    assert spec.preset == "2vcpu-8gb"
    assert spec.disk.size_bytes == 10 * 1024**3
    assert spec.container_command == "python"
    assert spec.args == "/supertaco/judge_job.py"
    names = [e.name for e in spec.environment_variables]
    assert names == ["TOKEN_FACTORY_BASE_URL", "TOKEN_FACTORY_API_KEY"]
    assert spec.environment_variables[0].value == "https://tf.example/v1"
    assert spec.environment_variables[1].mysterybox_secret.secret_id == "secret-1"
    assert spec.environment_variables[1].mysterybox_secret.version_id == "ver-1"
    paths = [f.container_path for f in spec.injected_files]
    assert paths == ["/supertaco/judge_job.py", "/supertaco/responses.json"]
    assert spec.injected_files[0].content == b"print(1)\n"


def test_build_job_spec_rejects_non_ascii_injection():
    bad = _request(files=[{"container_path": "/supertaco/x.py", "content": "héllo".encode()}])
    # content must be pure ASCII bytes: encode with ascii to pass, latin-1 to fail
    ok = _request(files=[{"container_path": "/supertaco/x.py", "content": b"plain"}])
    assert build_job_spec(ok).injected_files[0].content == b"plain"
    with pytest.raises(ValueError, match="ASCII"):
        build_job_spec(bad)


def test_state_name_normalizes_enum_and_string():
    class FakeEnum:
        name = "COMPLETED"

    assert _state_name(FakeEnum()) == "COMPLETED"
    assert _state_name("FAILED") == "FAILED"
    assert _state_name("JobStatus.State.ERROR") == "ERROR"


class _Op:
    def __init__(self, resource_id="aijob-1", status="done"):
        self.resource_id = resource_id
        self.status = status

    def wait(self):
        return self


class _Job:
    def __init__(self, state, details=""):
        self.status = type("S", (), {"state": state, "state_details": details})()

    def wait(self):
        return self


class FakeJobClient:
    """Scripted create/get/delete; records requests."""

    def __init__(self, states, details=""):
        self.states = list(states)
        self.details = details
        self.created = []
        self.deleted = []
        self.gets = 0

    def create(self, req):
        self.created.append(req)
        return _Op()

    def get(self, req):
        self.gets += 1
        state = self.states[min(self.gets - 1, len(self.states) - 1)]
        return _Job(state, self.details)

    def delete(self, req):
        self.deleted.append(req)
        return _Op()


def _client(fake, **kwargs):
    kwargs.setdefault("poll_interval_s", 0.0)
    kwargs.setdefault("timeout_s", 5.0)
    return NebiusJobsClient(project_id="project-1", job_client=fake, sleep=lambda s: None, **kwargs)


def test_run_completes_and_deletes():
    fake = FakeJobClient(["PROVISIONING", "RUNNING", "COMPLETED"])
    outcome = _client(fake).run(_request())
    assert outcome.success is True
    assert outcome.state == "COMPLETED"
    assert outcome.job_id == "aijob-1"
    assert len(fake.created) == 1
    assert len(fake.deleted) == 1  # deleted on success too


def test_run_failed_carries_state_details():
    fake = FakeJobClient(["FAILED"], details="disk full")
    outcome = _client(fake).run(_request())
    assert outcome.success is False
    assert outcome.state == "FAILED"
    assert "disk full" in (outcome.error or "")
    assert len(fake.deleted) == 1


def test_run_timeout_cancels_and_deletes():
    fake = FakeJobClient(["RUNNING"])  # never terminal
    client = NebiusJobsClient(
        project_id="project-1",
        job_client=fake,
        sleep=lambda s: None,
        poll_interval_s=0.0,
        timeout_s=0.0,  # deadline already passed after first poll check
    )
    outcome = client.run(_request())
    assert outcome.success is False
    assert "timeout" in (outcome.error or "").lower()
    assert len(fake.deleted) == 1


def test_run_create_error_still_returns_outcome():
    class Boom:
        def create(self, req):
            raise RuntimeError("PERMISSION_DENIED: nope")

    outcome = NebiusJobsClient(project_id="p", job_client=Boom(), sleep=lambda s: None).run(
        _request()
    )
    assert outcome.success is False
    assert "PERMISSION_DENIED" in (outcome.error or "")
    assert outcome.job_id is None


def test_run_emits_events():
    fake = FakeJobClient(["COMPLETED"])
    events = []
    _client(fake, on_event=lambda ev, **d: events.append((ev, d))).run(_request())
    kinds = [e[0] for e in events]
    assert "job_created" in kinds
    assert "job_completed" in kinds


def test_run_missing_auth_env_returns_error_outcome_not_raise(monkeypatch):
    # settings.py loads the real .env into os.environ on import, so force the
    # missing-auth path deterministically (and keep the test off the network).
    for key in ("NEBIUS_SERVICE_ACCOUNT_ID", "NEBIUS_PUBLIC_KEY_ID", "NEBIUS_PRIVATE_KEY_FILE"):
        monkeypatch.delenv(key, raising=False)
    outcome = NebiusJobsClient(project_id="p", job_client=None, sleep=lambda s: None).run(
        _request()
    )
    assert outcome.success is False
    assert outcome.job_id is None
    assert outcome.state == "ERROR"
    assert "NEBIUS" in (outcome.error or "")


def test_run_exception_after_create_still_deletes():
    class GetBoom:
        def __init__(self):
            self.calls = 0

        def create(self, req):
            return _Op()

        def get(self, req):
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("transient UNAVAILABLE")
            return _Job("RUNNING")

        def delete(self, req):
            self.deleted = getattr(self, "deleted", []) + [req]

    fake = GetBoom()
    outcome = NebiusJobsClient(
        project_id="p", job_client=fake, sleep=lambda s: None, poll_interval_s=0.0, timeout_s=5.0
    ).run(_request())
    assert outcome.success is False
    assert outcome.job_id == "aijob-1"  # id retained
    assert outcome.state == "ERROR"
    assert "transient UNAVAILABLE" in (outcome.error or "")
    assert len(fake.deleted) == 1  # deleted exactly once despite the get() failure


def test_run_delete_failure_does_not_raise_and_emits_job_delete_failed():
    class DeleteBoom:
        def __init__(self):
            self.deleted = 0

        def create(self, req):
            return _Op()

        def get(self, req):
            return _Job("COMPLETED")

        def delete(self, req):
            self.deleted += 1
            raise RuntimeError("delete PERMISSION_DENIED")

    events = []
    fake = DeleteBoom()
    outcome = NebiusJobsClient(
        project_id="p",
        job_client=fake,
        sleep=lambda s: None,
        poll_interval_s=0.0,
        timeout_s=5.0,
        on_event=lambda ev, **d: events.append((ev, d)),
    ).run(_request())
    assert outcome.success is True  # job itself completed; delete failure is a leak warning only
    assert outcome.state == "COMPLETED"
    assert fake.deleted == 1
    assert any(ev == "job_delete_failed" for ev, _ in events)


def test_run_raising_event_callback_does_not_break_run():
    fake = FakeJobClient(["COMPLETED"])

    def boom(ev, **d):
        raise RuntimeError("cb boom")

    outcome = _client(fake, on_event=boom).run(_request())
    assert outcome.success is True
    assert len(fake.deleted) == 1


def test_terminal_states_constant():
    assert TERMINAL_STATE_NAMES == frozenset({"COMPLETED", "FAILED", "CANCELLED", "ERROR"})


def test_job_outcome_defaults():
    o = JobOutcome(success=False, job_id=None, state="ERROR")
    assert o.error is None
