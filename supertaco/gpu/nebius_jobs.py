"""Nebius Serverless Job lifecycle: build spec, create, poll, always delete.

SDK imports happen inside functions (Windows cert shim lives in
_build_sdk_client; see docs/WS11_NEBIUS_SPIKE.md). No eval knowledge here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

TERMINAL_STATE_NAMES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "ERROR"})


@dataclass
class JobOutcome:
    """Result of one job run: success flag plus terminal state and error."""

    success: bool
    job_id: Optional[str]
    state: str
    error: Optional[str] = None
    state_details: str = ""


@dataclass
class JobRequest:
    """Plain-data job description; build_job_spec maps it to the SDK JobSpec."""

    name: str
    image: str
    platform: str
    preset: str
    disk_gib: int
    command: str
    args: str
    env: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)


def _state_name(state: Any) -> str:
    """Normalize JobStatus.State enum (or a plain string) to its bare name."""
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    text = str(state)
    return text.rsplit(".", 1)[-1]


def build_job_spec(req: JobRequest, project_id: str) -> Any:
    """Assemble an SDK JobSpec; injections must be pure ASCII (cp1252-safe)."""
    from nebius.api.nebius.ai.v1 import JobSpec
    from nebius.api.nebius.compute.v1 import DiskSpec as ComputeDiskSpec

    env_vars = []
    for entry in req.env:
        if "value" in entry:
            env_vars.append(JobSpec.EnvironmentVariable(name=entry["name"], value=entry["value"]))
        else:
            ref = entry["mysterybox"]
            env_vars.append(
                JobSpec.EnvironmentVariable(
                    name=entry["name"],
                    mysterybox_secret=JobSpec.MysteryBoxSecretRef(
                        secret_id=ref["secret_id"], version_id=ref["version_id"]
                    ),
                )
            )
    injected = []
    for f in req.files:
        content = f["content"]
        try:
            content.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"injected file {f['container_path']} must be pure ASCII") from exc
        injected.append(JobSpec.FileInjection(container_path=f["container_path"], content=content))
    return JobSpec(
        image=req.image,
        container_command=req.command,
        args=req.args,
        platform=req.platform,
        preset=req.preset,
        disk=JobSpec.DiskSpec(
            type=ComputeDiskSpec.DiskType.NETWORK_SSD, size_bytes=req.disk_gib * 1024**3
        ),
        environment_variables=env_vars,
        injected_files=injected,
    )


def _build_sdk_client(project_id: str, user_agent_prefix: str) -> Any:
    """Real SDK JobServiceClient (authorized-key auth + certifi TLS shim)."""
    import os
    import pathlib

    import certifi
    from grpc import ssl_channel_credentials  # type: ignore[import-untyped]  # grpc ships no stubs
    from nebius.api.nebius.ai.v1 import JobServiceClient
    from nebius.sdk import SDK

    missing = [
        key
        for key in (
            "NEBIUS_SERVICE_ACCOUNT_ID",
            "NEBIUS_PUBLIC_KEY_ID",
            "NEBIUS_PRIVATE_KEY_FILE",
        )
        if not os.environ.get(key)
    ]
    if missing:
        raise ValueError(f"missing Nebius auth env: {', '.join(missing)}; set them in .env")
    with open(certifi.where(), "rb") as fh:
        ca = fh.read()
    sdk = SDK(
        service_account_id=os.environ["NEBIUS_SERVICE_ACCOUNT_ID"],
        service_account_public_key_id=os.environ["NEBIUS_PUBLIC_KEY_ID"],
        service_account_private_key_file_name=str(
            pathlib.Path(os.environ["NEBIUS_PRIVATE_KEY_FILE"]).expanduser()
        ),
        parent_id=project_id,
        user_agent_prefix=user_agent_prefix,
        tls_credentials=ssl_channel_credentials(root_certificates=ca),
    )
    return JobServiceClient(sdk)


class NebiusJobsClient:
    """Create → poll → delete a Nebius job; timeouts cancel+delete; never raises."""

    def __init__(
        self,
        project_id: str,
        *,
        job_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_s: float = 10.0,
        timeout_s: float = 900.0,
        user_agent_prefix: str = "supertaco-ws1.2/0.5",
        on_event: Optional[Callable[..., None]] = None,
    ) -> None:
        self.project_id = project_id
        self._jobs = job_client
        self._sleep = sleep
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s
        self._user_agent_prefix = user_agent_prefix
        self._on_event = on_event

    def _client(self) -> Any:
        if self._jobs is None:
            self._jobs = _build_sdk_client(self.project_id, self._user_agent_prefix)
        return self._jobs

    def _emit(self, event: str, **data: Any) -> None:
        if self._on_event is not None:
            self._on_event(event, **data)

    def run(self, req: JobRequest) -> JobOutcome:
        """Run to a terminal state; the job resource is always deleted."""
        from nebius.api.nebius.ai.v1 import CreateJobRequest, GetJobRequest
        from nebius.api.nebius.common.v1 import ResourceMetadata

        client = self._client()
        job_id: Optional[str] = None
        try:
            spec = build_job_spec(req, self.project_id)
            op = client.create(
                CreateJobRequest(
                    metadata=ResourceMetadata(parent_id=self.project_id, name=req.name),
                    spec=spec,
                )
            ).wait()
            job_id = getattr(op, "resource_id", None)
            if not job_id:
                return JobOutcome(
                    False, None, "ERROR", error=f"create returned no resource id: {op}"
                )
            self._emit("job_created", job_id=job_id, name=req.name)
            deadline = time.monotonic() + self._timeout_s
            state = "UNKNOWN"
            details = ""
            while True:
                job = client.get(GetJobRequest(id=job_id)).wait()
                status = getattr(job, "status", None)
                state = _state_name(getattr(status, "state", "UNKNOWN")) if status else "UNKNOWN"
                details = str(getattr(status, "state_details", "") or "") if status else ""
                self._emit("job_state", job_id=job_id, state=state)
                if state in TERMINAL_STATE_NAMES:
                    break
                if time.monotonic() >= deadline:
                    self._delete(client, job_id)
                    return JobOutcome(
                        False,
                        job_id,
                        state,
                        error=f"job {job_id} timeout after {self._timeout_s:.0f}s in {state}",
                        state_details=details,
                    )
                self._sleep(self._poll_interval_s)
            self._delete(client, job_id)
            if state == "COMPLETED":
                self._emit("job_completed", job_id=job_id)
                return JobOutcome(True, job_id, state, state_details=details)
            return JobOutcome(
                False,
                job_id,
                state,
                error=f"job {job_id} ended {state}: {details or 'no state_details'}",
                state_details=details,
            )
        except Exception as exc:  # infra failure surfaces as a labeled outcome
            if job_id:
                self._delete(client, job_id)
            return JobOutcome(False, job_id, "ERROR", error=f"{type(exc).__name__}: {exc}")

    def _delete(self, client: Any, job_id: str) -> None:
        from nebius.api.nebius.ai.v1 import DeleteJobRequest

        try:
            client.delete(DeleteJobRequest(id=job_id)).wait()
            self._emit("job_deleted", job_id=job_id)
        except Exception as exc:
            self._emit("job_delete_failed", job_id=job_id, error=str(exc))
