import json
import os
from datetime import datetime


class ObjectStorage:
    """Helper for log/checkpoint object storage on Nebius."""

    def __init__(self, base_path: str = "logs"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_log(self, job_id: str, step: str, content: str) -> str:
        """Save a log entry for a job."""
        filepath = os.path.join(self.base_path, f"{job_id}_{step}_{datetime.utcnow().isoformat()}.jsonl")
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def save_checkpoint(self, job_id: str, config: dict, metadata: dict = None) -> str:
        """Save a training config as a checkpoint artifact."""
        filepath = os.path.join(self.base_path, f"{job_id}_config_{datetime.utcnow().isoformat()}.yaml")
        with open(filepath, "w") as f:
            f.write(self._config_to_yaml(config, metadata))
        return filepath

    def _config_to_yaml(self, config: dict, metadata: dict = None) -> str:
        """Convert config dict to YAML (2-space indent, alphabetized keys)."""
        lines = []
        if metadata:
            for k, v in sorted(metadata.items()):
                lines.append(f"{k}: {v}")
        for k, v in sorted(config.items()):
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sub_k, sub_v in sorted(v.items()):
                    lines.append(f"  {sub_k}: {sub_v}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)