"""SuperTaco CLI entry point.

Usage:
    supertaco run <config> [--dry-run]
    supertaco --help
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from supertaco.agent.llm import NemotronClient
from supertaco.errors import ConfigurationError
from supertaco.runner import run as runner_run
from supertaco.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="SuperTaco: AI-supervised fine-tuning sandbox")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a fine-tuning job")
    run_parser.add_argument("config", help="Path to YAML config file")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without launching real jobs"
    )

    args = parser.parse_args()
    if args.command == "run":
        run_config(args.config, args.dry_run)
    else:
        parser.print_help()
        sys.exit(0)


def _make_llm() -> NemotronClient:
    return NemotronClient(
        base_url=settings.token_factory_base_url,
        api_key=settings.nebius_api_key.get_secret_value(),
    )


def run_config(config_path: str, dry_run: bool = True) -> None:
    """Run a fine-tuning job from a config file via the shared runner."""
    try:
        config = _load_config(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error loading {config_path}: {exc}") from exc
    print(f"SuperTaco: running {config_path}")
    llm = _make_llm()

    def printer(event) -> None:
        print(f"[{event.type}] {json.dumps(event.data, default=str)}")

    try:
        result = runner_run(config, max_retries=3, dry_run=dry_run, on_event=printer, llm=llm)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    if result.success:
        print(f"Success after {result.attempts} attempt(s). Configs: {result.configs_written}")
    else:
        print(f"Run failed: {result.error}")
        sys.exit(1)


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config or {}


if __name__ == "__main__":
    main()
