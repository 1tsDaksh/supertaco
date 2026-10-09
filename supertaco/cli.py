"""SuperTaco CLI entry point.

Usage:
    supertaco run <config>
    supertaco --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from supertaco.errors import ConfigurationError
from supertaco.loop import make_llm, run_training_loop
from supertaco.planner import materialize_dataset, plan_run, plan_to_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SuperTaco: AI-supervised fine-tuning sandbox")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Fine-tune on a Colab T4 with the supervisor loop"
    )
    run_parser.add_argument("config", help="Path to YAML config file")

    plan_parser = subparsers.add_parser(
        "plan", help="Turn a free-text goal into a reviewable run plan YAML"
    )
    plan_parser.add_argument("goal", help="What you want the model to learn")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_config(args.config)
    elif args.command == "plan":
        plan_goal_command(args.goal)
    else:
        parser.print_help()
        sys.exit(0)


def plan_goal_command(goal: str, llm=None) -> None:
    """Turn a free-text goal into configs/runs/plan_<ts>.yaml for review (G7)."""
    from datetime import UTC, datetime

    llm = llm if llm is not None else make_llm()
    plan = materialize_dataset(plan_run(goal, llm), goal, llm)
    config = plan_to_config(plan)
    config["goal"] = goal
    config["eval_prompts"] = list(plan.eval_prompts)
    runs_dir = Path("configs/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = runs_dir / f"plan_{stamp}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True, allow_unicode=True)
    if plan.dataset.kind == "synthetic":
        source = f"synthetic ({len(plan.dataset.rows or [])} rows)"
    else:
        source = f"hf:{plan.dataset.hf_id}"
    print(f"Plan written: {path} (mode={plan.plan_mode}, dataset={source})")
    if plan.plan_mode == "fallback":
        print("warning: planner fell back to shipped defaults - review before running")
    if plan.dataset.note:
        print(f"note: {plan.dataset.note}")
    print(f"Run it with: supertaco run {path}")


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
    eval_prompts = config.pop("eval_prompts", None) if isinstance(config, dict) else None

    def printer(event) -> None:
        print(f"[{event.type}] {json.dumps(event.data, default=str)}")

    try:
        result = run_training_loop(
            config, transport=transport, llm=llm, on_event=printer, prompts=eval_prompts
        )
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
