"""Judge-as-a-Nebius-Job backend: MysteryBox key, S3 presign, job, audit merge.

Raises JudgeBackendError on any infra failure; the loop converts that into a
labeled direct-judging fallback. SDK/boto3 imports stay inside methods so
construction validation never touches the network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from supertaco.errors import JudgeBackendError
from supertaco.gpu.nebius_jobs import JobOutcome, JobRequest, NebiusJobsClient
from supertaco.paths import repo_root

JUDGE_SCRIPT_PATH = Path(__file__).resolve().parent / "judge_job.py"
SECRET_NAME = "supertaco-tf-key"
BUCKET_NAME = "supertaco-judge-artifacts"
PRESIGN_TTL_SECONDS = 3600


def _append_audit(entries: list[dict]) -> None:
    """Append judge-job call entries to logs/llm_calls.jsonl (warning only)."""
    import warnings

    try:
        log_dir = repo_root() / "logs"
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "llm_calls.jsonl", "a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        warnings.warn(f"llm_calls.jsonl write failed: {exc}", stacklevel=2)


class _S3Artifacts:
    """boto3-backed bucket ensure / presign / download / delete."""

    def __init__(self, bucket_name: str = BUCKET_NAME) -> None:
        self._bucket_name = bucket_name
        self._client: Any = None
        self._endpoint: Optional[str] = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]

        access_key = _require_env("NEBIUS_S3_ACCESS_KEY_ID")
        secret_key = _require_env("NEBIUS_S3_SECRET_KEY")
        endpoint = _require_env("NEBIUS_S3_ENDPOINT")
        self._endpoint = endpoint
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )
        return self._client

    def ensure_bucket(self) -> str:
        client = self._ensure_client()
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]  # no stubs

        try:
            client.head_bucket(Bucket=self._bucket_name)
        except ClientError:
            try:
                client.create_bucket(Bucket=self._bucket_name)
            except ClientError as exc:
                raise JudgeBackendError(
                    f"cannot create judge bucket {self._bucket_name}: {exc}"
                ) from exc
        return self._bucket_name

    def presign_put(self, key: str) -> str:
        client = self._ensure_client()
        return client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket_name, "Key": key},
            ExpiresIn=PRESIGN_TTL_SECONDS,
        )

    def download_json(self, key: str) -> dict:
        client = self._ensure_client()
        try:
            obj = client.get_object(Bucket=self._bucket_name, Key=key)
            return json.loads(obj["Body"].read().decode())
        except Exception as exc:  # ClientError/BotoCoreError/JSONDecodeError/UnicodeDecodeError
            raise JudgeBackendError(f"cannot read scores object {key}: {exc}") from exc

    def delete_object(self, key: str) -> None:
        try:
            client = self._ensure_client()
            client.delete_object(Bucket=self._bucket_name, Key=key)
        except Exception:
            pass  # best-effort cleanup


class _MysteryBoxSecrets:
    """Ensure the TF key exists as a MysteryBox secret + primary version."""

    def ensure_tf_secret(self, project_id: str, name: str, tf_key: str) -> tuple[str, str]:
        from nebius.api.nebius.common.v1 import ResourceMetadata
        from nebius.api.nebius.mysterybox.v1 import (
            CreateSecretRequest,
            CreateSecretVersionRequest,
            GetSecretByNameRequest,
            Payload,
            SecretServiceClient,
            SecretSpec,
            SecretVersionServiceClient,
            SecretVersionSpec,
        )

        sdk = _build_mysterybox_sdk(project_id)
        secrets = SecretServiceClient(sdk)
        versions = SecretVersionServiceClient(sdk)
        try:
            secret = secrets.get_by_name(
                GetSecretByNameRequest(parent_id=project_id, name=name)
            ).wait()
            secret_id = secret.metadata.id
            primary = (secret.spec.primary_version_id if secret.spec else "") or ""
            if primary:
                return secret_id, primary
        except Exception:
            secret_id = None
        if secret_id is None:
            op = secrets.create(
                CreateSecretRequest(
                    metadata=ResourceMetadata(parent_id=project_id, name=name),
                    spec=SecretSpec(description="SuperTaco Token Factory judge key"),
                )
            ).wait()
            secret_id = getattr(op, "resource_id", None)
            if not secret_id:
                raise JudgeBackendError(f"MysteryBox secret create returned no id: {op}")
        vop = versions.create(
            CreateSecretVersionRequest(
                metadata=ResourceMetadata(parent_id=secret_id, name=f"v{int(time.time())}"),
                spec=SecretVersionSpec(
                    payload=[Payload(key="TOKEN_FACTORY_API_KEY", string_value=tf_key)],
                    set_primary=True,
                ),
            )
        ).wait()
        version_id = getattr(vop, "resource_id", None)
        if not version_id:
            raise JudgeBackendError(f"MysteryBox version create returned no id: {vop}")
        return secret_id, version_id


def _require_env(key: str) -> str:
    import os

    value = os.environ.get(key, "")
    if not value:
        raise JudgeBackendError(f"missing {key}; set it in .env")
    return value


def _build_mysterybox_sdk(project_id: str) -> Any:
    import os
    import pathlib

    import certifi
    from grpc import ssl_channel_credentials  # type: ignore[import-untyped]  # grpc ships no stubs
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
        raise JudgeBackendError(f"missing Nebius auth env: {', '.join(missing)}; set them in .env")
    with open(certifi.where(), "rb") as fh:
        ca = fh.read()
    return SDK(
        service_account_id=os.environ["NEBIUS_SERVICE_ACCOUNT_ID"],
        service_account_public_key_id=os.environ["NEBIUS_PUBLIC_KEY_ID"],
        service_account_private_key_file_name=str(
            pathlib.Path(os.environ["NEBIUS_PRIVATE_KEY_FILE"]).expanduser()
        ),
        parent_id=project_id,
        user_agent_prefix="supertaco-ws1.2-mysterybox/0.5",
        tls_credentials=ssl_channel_credentials(root_certificates=ca),
    )


class NebiusJudgeBackend:
    """Batch judge(prompts, base, ft) -> run_eval_suite-shaped dict.

    Infra failures raise JudgeBackendError (loop falls back to direct judging).
    """

    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        tf_base_url: Optional[str] = None,
        tf_api_key: Optional[str] = None,
        jobs: Any = None,
        artifacts: Any = None,
        secrets: Any = None,
        job_timeout_s: float = 900.0,
        on_event: Optional[Any] = None,
    ) -> None:
        import os

        from supertaco.settings import settings

        self.project_id = (
            project_id or os.environ.get("NEBIUS_PROJECT_ID", "") or settings.nebius_project_id
        )
        if not self.project_id:
            raise JudgeBackendError("missing NEBIUS_PROJECT_ID; set it in .env")
        self.tf_base_url = tf_base_url or settings.token_factory_base_url
        self.tf_api_key = tf_api_key or settings.nebius_api_key.get_secret_value()
        if not self.tf_api_key:
            raise JudgeBackendError("missing NEBIUS_API_KEY (Token Factory key); set it in .env")
        self._jobs = (
            jobs
            if jobs is not None
            else NebiusJobsClient(self.project_id, timeout_s=job_timeout_s, on_event=on_event)
        )
        self._artifacts = artifacts if artifacts is not None else _S3Artifacts()
        self._secrets = secrets if secrets is not None else _MysteryBoxSecrets()
        self._on_event = on_event

    def judge(self, prompts: list[str], base_responses: dict, fine_tuned_responses: dict) -> dict:
        """Score the suite in one Nebius job; raise JudgeBackendError on any failure."""
        try:
            return self._judge_inner(prompts, base_responses, fine_tuned_responses)
        except JudgeBackendError:
            raise
        except Exception as exc:
            raise JudgeBackendError(f"judge failed: {type(exc).__name__}: {exc}") from exc

    def _judge_inner(
        self, prompts: list[str], base_responses: dict, fine_tuned_responses: dict
    ) -> dict:
        """Run the job pipeline; return harness-shaped batch dict."""
        try:
            script = JUDGE_SCRIPT_PATH.read_bytes()
        except OSError as exc:
            raise JudgeBackendError(f"cannot read judge script: {exc}") from exc
        try:
            responses = {
                "prompts": list(prompts),
                "base": {p: base_responses[p] for p in prompts},
                "fine_tuned": {p: fine_tuned_responses[p] for p in prompts},
            }
        except KeyError as exc:
            raise JudgeBackendError(f"responses payload missing prompt key: {exc}") from exc
        try:
            secret_id, version_id = self._secrets.ensure_tf_secret(
                self.project_id, SECRET_NAME, self.tf_api_key
            )
            self._artifacts.ensure_bucket()
            key = f"scores-{int(time.time())}.json"
            put_url = self._artifacts.presign_put(key)
        except JudgeBackendError:
            raise
        except Exception as exc:
            raise JudgeBackendError(f"judge job setup failed: {type(exc).__name__}: {exc}") from exc

        req = JobRequest(
            name=f"supertaco-judge-{int(time.time())}",
            image="python:3.12-slim",
            platform="cpu-d3",
            preset="2vcpu-8gb",
            disk_gib=10,
            command="python",
            args="/supertaco/judge_job.py",
            env=[
                {"name": "TOKEN_FACTORY_BASE_URL", "value": self.tf_base_url},
                {
                    "name": "TOKEN_FACTORY_API_KEY",
                    "mysterybox": {"secret_id": secret_id, "version_id": version_id},
                },
                {"name": "SCORES_PUT_URL", "value": put_url},
                {"name": "RESPONSES_PATH", "value": "/supertaco/responses.json"},
            ],
            files=[
                {"container_path": "/supertaco/judge_job.py", "content": script},
                {
                    "container_path": "/supertaco/responses.json",
                    "content": json.dumps(responses).encode("utf-8"),
                },
            ],
        )
        try:
            outcome: JobOutcome = self._jobs.run(req)
            if not outcome.success:
                raise JudgeBackendError(outcome.error or f"judge job ended {outcome.state}")
            payload = self._artifacts.download_json(key)
            batch = self._validate_batch(payload, len(prompts))
        finally:
            try:
                self._artifacts.delete_object(key)
            except Exception:
                pass
        stamp = time.time()
        call_log = [
            {
                "model_key": "judge",
                "model_id": "nvidia/nemotron-3-super-120b-a12b",
                "mode": entry.get("mode", "real"),
                "tokens": entry.get("tokens", 0),
                "latency": entry.get("latency", 0.0),
                "error": entry.get("error"),
                "timestamp": stamp,
                "response_preview": str(entry.get("response_preview") or "")[:200],
                "judge_backend": "nebius_job",
                "job_id": outcome.job_id,
            }
            for entry in batch.get("call_log", [])
            if isinstance(entry, dict)
        ]
        _append_audit(call_log)
        return {
            "prompts": batch["prompts"],
            "base_scores": batch["base_scores"],
            "fine_tuned_scores": batch["fine_tuned_scores"],
            "modes": batch["modes"],
            "call_log": call_log,
            "judge_backend": "nebius_job",
        }

    @staticmethod
    def _validate_batch(payload: Any, expected: int) -> dict:
        if not isinstance(payload, dict):
            raise JudgeBackendError(f"scores payload is not an object: {payload!r:.200}")
        for list_key in ("prompts", "base_scores", "fine_tuned_scores", "modes"):
            if not isinstance(payload.get(list_key), list):
                raise JudgeBackendError(f"scores payload missing list key {list_key!r}")
            if len(payload[list_key]) != expected:
                raise JudgeBackendError(
                    f"scores payload {list_key} has {len(payload[list_key])} entries,"
                    f" expected {expected}"
                )
        for score in payload["base_scores"] + payload["fine_tuned_scores"]:
            if not isinstance(score, (int, float)):
                raise JudgeBackendError(f"scores payload has non-numeric score {score!r}")
        return payload
