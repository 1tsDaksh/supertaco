"""SuperTaco CLI entry point.

Usage:
    supertaco run <config>
    supertaco --help
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from supertaco.errors import ConfigurationError
from supertaco.loop import make_llm, run_training_loop
from supertaco.planner import materialize_dataset, plan_run, plan_to_config

RUNS_DIR = Path("configs/runs")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SuperTaco: AI-supervised fine-tuning sandbox")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Fine-tune on a Colab T4 with the supervisor loop"
    )
    run_parser.add_argument("config", help="Path to YAML config file")
    run_parser.add_argument("--feedback", help="Revise the plan from free-text feedback, then run")

    plan_parser = subparsers.add_parser(
        "plan", help="Turn a free-text goal into a reviewable run plan YAML"
    )
    plan_parser.add_argument("goal", help="What you want the model to learn")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_config(args.config, feedback=args.feedback)
    elif args.command == "plan":
        plan_goal_command(args.goal)
    else:
        parser.print_help()
        sys.exit(0)


def _write_plan_yaml(config: dict, suffix: str) -> Path:
    """Write a plan yaml to RUNS_DIR with a timestamped name (G7)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = RUNS_DIR / f"plan_{stamp}{suffix}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True, allow_unicode=True)
    return path


def plan_goal_command(goal: str, llm=None) -> None:
    """Turn a free-text goal into configs/runs/plan_<ts>.yaml for review (G7)."""
    llm = llm if llm is not None else make_llm()
    plan = materialize_dataset(plan_run(goal, llm), goal, llm)
    config = plan_to_config(plan)
    config["goal"] = goal
    config["eval_prompts"] = list(plan.eval_prompts)
    path = _write_plan_yaml(config, "")
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


def run_config(config_path: str, llm=None, transport=None, feedback: str | None = None) -> None:
    """Run the real supervisor loop for a config file.

    llm/transport default to the Token Factory client and the Colab
    transport; tests inject fakes through these parameters or by
    monkeypatching _make_transport. With feedback, the plan is revised
    (Nemotron), written as *_revised.yaml, then trained.
    """
    try:
        config = _load_config(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error loading {config_path}: {exc}") from exc
    print(f"SuperTaco: running {config_path}")
    llm = llm if llm is not None else make_llm()
    transport = transport if transport is not None else _make_transport()
    if feedback:
        goal = config.get("goal") if isinstance(config, dict) else None
        if not isinstance(goal, str) or not goal.strip():
            raise SystemExit(
                "--feedback needs a config produced by `supertaco plan` (missing goal key)"
            )
        from supertaco.planner import revise_run_config

        config, revised = revise_run_config(feedback, config, goal.strip(), llm)
        path = _write_plan_yaml(config, "_revised")
        print(f"Revised plan written: {path} (mode={revised.plan_mode})")
        if revised.revision_note:
            print(f"note: {revised.revision_note}")
        if revised.plan_mode == "fallback":
            print("warning: revision fell back to the previous plan")
    eval_prompts = config.pop("eval_prompts", None) if isinstance(config, dict) else None
    if isinstance(config, dict):
        config.pop("goal", None)
        config.pop("revision_note", None)
    if eval_prompts is not None and (
        not isinstance(eval_prompts, list)
        or not all(isinstance(p, str) and p.strip() for p in eval_prompts)
    ):
        eval_prompts = None

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
