"""SuperTaco CLI entry point.

Usage:
    supertaco run <config>
    supertaco --help
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from supertaco.errors import ConfigurationError
from supertaco.loop import make_llm, run_training_loop


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SuperTaco: AI-supervised fine-tuning sandbox")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Fine-tune on a Colab T4 with the supervisor loop"
    )
    run_parser.add_argument("config", help="Path to YAML config file")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_config(args.config)
    else:
        parser.print_help()
        sys.exit(0)


def _make_transport():
    from supertaco.gpu.transport import ColabTransport

    return ColabTransport()


def run_config(config_path: str, llm=None, transport=None) -> None:
    """Run the real supervisor loop for a config file.

    llm/transport default to the Token Factory client and the Colab
    transport; tests inject fakes through these parameters or by
    monkeypatching _make_transport.
    """
    try:
        config = _load_config(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error loading {config_path}: {exc}") from exc
    print(f"SuperTaco: running {config_path}")
    llm = llm if llm is not None else make_llm()
    transport = transport if transport is not None else _make_transport()

    def printer(event) -> None:
        print(f"[{event.type}] {json.dumps(event.data, default=str)}")

    try:
        result = run_training_loop(config, transport=transport, llm=llm, on_event=printer)
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
