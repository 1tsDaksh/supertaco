"""SandboxTune CLI entry point.

Usage:
    supertaco run <config> [--dry-run]
    supertaco --help
"""

import argparse
import sys
import os
from pathlib import Path

from supertaco.settings import SandboxTuneSettings
from supertaco.nebius.jobs import NebiusJobClient
from supertaco.nebius.endpoints import NebiusEndpointClient
from supertaco.agent.graph import run_agent_graph
from supertaco.errors import MaxRetriesExceeded


def main():
    parser = argparse.ArgumentParser(
        description="SandboxTune: AI-supervised fine-tuning sandbox"
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a fine-tuning job")
    run_parser.add_argument("config", help="Path to YAML config file")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without launching real jobs",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_config(args.config, args.dry_run)
    else:
        parser.print_help()
        sys.exit(0)


def run_config(config_path: str, dry_run: bool = True):
    """Run a fine-tuning job from a config file."""
    # Load settings
    settings = SandboxTuneSettings()

    # Initialize clients
    job_client = NebiusJobClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key,
        project_id=settings.nebius_project_id,
    )

    endpoint_client = NebiusEndpointClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key,
    )

    llm = NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key,
    )

    # Load config
    config = _load_config(config_path)

    print(f"SandboxTune: Running config from {config_path}")
    print(f"Environment: {settings.environment}")
    print(f"Config keys: {list(config.keys())}")

    if dry_run:
        print("[dry-run mode] Would launch job without real API calls")

    try:
        result = run_agent_loop(
            initial_config=config,
            job_client=job_client,
            endpoint_client=endpoint_client,
            llm=llm,
            max_retries=3,
            dry_run=dry_run,
        )
        print(f"Job completed successfully!")
        print(f"Final config: {result['config']}")
        print(f"Job ID: {result['job_id']}")
        print(f"Attempts: {result['attempts']}")

    except MaxRetriesExceeded as e:
        print(f"Job failed after max retries: {e}")
        print(f"Last config attempted: {e.last_config}")
        sys.exit(1)


def _load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


if __name__ == "__main__":
    main()