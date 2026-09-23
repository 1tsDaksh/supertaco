"""SandboxTune settings - loaded from .env file or environment variables."""

import os
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr


def load_env_file(path: str = ".env"):
    """Load environment variables from .env file."""
    env_path = Path(path)
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already set
                    if key not in os.environ:
                        os.environ[key] = value


# Load .env file on import
load_env_file()


class SandboxTuneSettings(BaseModel):
    """Settings model for SandboxTune.
    
    Fields read from environment variables (set by .env file on import).
    """
    
    nebius_api_key: SecretStr = Field(
        default="",
        description="Nebius AI Cloud / Token Factory API key"
    )
    nebius_project_id: str = Field(
        default="",
        description="Nebius project ID"
    )
    tavily_api_key: SecretStr = Field(
        default="",
        description="Tavily API key for error context lookup"
    )
    token_factory_base_url: str = Field(
        default="https://api.token.factory.nvidia.com/v1",
        description="Token Factory base URL (OpenAI-compatible)"
    )
    environment: str = Field(
        default="dev",
        description="Environment: dev | prod"
    )
    
    model_config = {"extra": "forbid"}
    
    def __init__(self, **data):
        # Extract values from os.environ if not provided
        env_values = {}
        for field_name in ["nebius_api_key", "nebius_project_id", "tavily_api_key"]:
            env_key = field_name.upper()
            if env_key in os.environ and not data.get(field_name):
                env_values[field_name] = os.environ[env_key]
        
        # Also check for TOKEN_FACTORY_BASE_URL and SANDBOXTUNE_ENV
        if "TOKEN_FACTORY_BASE_URL" in os.environ and not data.get("token_factory_base_url"):
            env_values["token_factory_base_url"] = os.environ["TOKEN_FACTORY_BASE_URL"]
        if "SANDBOXTUNE_ENV" in os.environ and not data.get("environment"):
            env_values["environment"] = os.environ["SANDBOXTUNE_ENV"]
        
        # Update data with env values
        data.update(env_values)
        super().__init__(**data)
        
        # Ensure environment is valid
        if self.environment not in ("dev", "prod"):
            self.environment = "dev"


# Create global settings instance
settings = SandboxTuneSettings()