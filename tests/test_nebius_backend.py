"""Nebius judge backend: env validation, orchestration, labeled failures."""

from supertaco.errors import JudgeBackendError


def test_judge_backend_error_is_exception_with_message():
    err = JudgeBackendError("missing NEBIUS_PROJECT_ID; set it in .env")
    assert isinstance(err, Exception)
    assert "NEBIUS_PROJECT_ID" in str(err)
