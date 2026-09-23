import httpx
from sandboxtune.errors import JobFailedError


class NebiusJobClient:
    """Wrapper for Nebius AI Cloud Serverless Jobs SDK."""

    def __init__(self, base_url: str, api_key: str, project_id: str):
        self.base_url = base_url
        self.api_key = api_key
        self.project_id = project_id
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def launch_job(self, config: dict, dry_run: bool = False) -> dict:
        """Launch a Serverless Job. If dry_run, returns the payload that would be sent."""
        payload = {
            "projectId": self.project_id,
            "config": config,
        }
        if dry_run:
            return {"dry_run": True, "payload": payload, "job_id": None}
        # Real API call would go here
        raise NotImplementedError("Real Nebius API call - use dry_run=True for now")

    def get_status(self, job_id: str) -> dict:
        """Get job status by ID."""
        raise NotImplementedError

    def stop_job(self, job_id: str) -> None:
        """Stop a running job."""
        raise NotImplementedError