import httpx


class NebiusEndpointClient:
    """Wrapper for Nebius AI Cloud Serverless Endpoints SDK."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def deploy(self, checkpoint_path: str, model_name: str) -> dict:
        """Deploy a checkpoint to a Serverless Endpoint."""
        payload = {
            "checkpointPath": checkpoint_path,
            "modelName": model_name,
        }
        return {"dry_run": True, "payload": payload, "endpoint_id": None}

    def undeploy(self, endpoint_id: str) -> None:
        """Undeploy a Serverless Endpoint."""
        pass

    def get_status(self, endpoint_id: str) -> dict:
        """Get endpoint status."""
        raise NotImplementedError