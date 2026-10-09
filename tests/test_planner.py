"""Goal -> RunPlan: parse, G10 clamps, fallbacks, dataset materialization."""

import json

import pytest

from supertaco.data import generate_dataset  # noqa: F401  (import sanity: planner uses it)
from supertaco.eval.harness import DEFAULT_PROMPTS
from supertaco.planner import (
    RunPlan,  # noqa: F401  (import sanity: planner exports it)
    default_plan,
    materialize_dataset,
    plan_run,
    plan_to_config,
)

GOOD = {
    "model": "qwen2.5-1.5b",
    "learning_rate": 5e-4,
    "batch_size": 8,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "num_epochs": 2,
    "dataset": {"kind": "synthetic"},
    "eval_prompts": [f"goal prompt {i}" for i in range(5)],
}


class _LLM:
    def __init__(self, reply):
        self.reply = reply
        self.goals = []

    def plan_goal(self, goal):
        self.goals.append(goal)
        return self.reply


def test_plan_run_parses_valid_json():
    plan = plan_run("pirate chef", _LLM(json.dumps(GOOD)))
    assert plan.plan_mode == "real"
    assert plan.model == "Qwen/Qwen2.5-1.5B-Instruct"  # canonicalized via whitelist
    assert plan.learning_rate == pytest.approx(5e-4)
    assert plan.dataset.kind == "synthetic"
    assert plan.eval_prompts == [f"goal prompt {i}" for i in range(5)]


def test_plan_run_clamps_and_fixes_bad_values():
    bad = dict(
        GOOD,
        model="llama-3-8b",
        learning_rate=5.0,
        num_epochs=9,
        batch_size=99,
        eval_prompts=["only", "three"],
    )
    plan = plan_run("x", _LLM(json.dumps(bad)))
    assert plan.model == "qwen2.5-0.5b"  # not whitelisted -> default
    assert plan.learning_rate == pytest.approx(1e-3)  # clamped to upper bound
    assert plan.num_epochs == 3
    assert plan.batch_size == 16
    assert plan.eval_prompts == DEFAULT_PROMPTS  # wrong count -> defaults


def test_plan_run_falls_back_on_garbage_reply():
    plan = plan_run("x", _LLM("I cannot help with that."))
    assert plan.plan_mode == "fallback"
    assert plan.model == "qwen2.5-0.5b"
    assert plan.dataset.hf_id == "yahma/alpaca-cleaned"
    assert plan.eval_prompts == DEFAULT_PROMPTS


def test_plan_run_falls_back_when_llm_raises():
    class Boom:
        def plan_goal(self, goal):
            raise RuntimeError("token factory down")

    plan = plan_run("x", Boom())
    assert plan.plan_mode == "fallback"


def test_plan_to_config_contains_knobs_and_dataset_not_prompts():
    plan = plan_run("x", _LLM(json.dumps(GOOD)))
    cfg = plan_to_config(plan)
    assert cfg["learning_rate"] == pytest.approx(5e-4)
    assert cfg["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["dataset"]["kind"] == "synthetic"
    assert "rows" not in cfg["dataset"]  # rows arrive via materialize_dataset
    assert "eval_prompts" not in cfg and "plan_mode" not in cfg


def test_default_plan_is_flagged_fallback():
    plan = default_plan()
    assert plan.plan_mode == "fallback"
    assert plan.eval_prompts == DEFAULT_PROMPTS


def test_materialize_dataset_fills_synthetic_rows():
    plan = plan_run("x", _LLM(json.dumps(GOOD)))

    class Gen:
        def __init__(self):
            self.n = 0

        def generate_rows(self, goal, count, banned):
            start = self.n
            rows = [
                {"instruction": f"row {n}", "input": "", "output": "a" * 10}
                for n in range(start, start + count)
            ]
            self.n = start + count
            return json.dumps(rows)

    materialize_dataset(plan, "x", Gen())
    assert plan.dataset.kind == "synthetic"
    assert len(plan.dataset.rows) == 150


def test_materialize_dataset_falls_back_to_alpaca_when_empty():
    plan = plan_run("x", _LLM(json.dumps(GOOD)))

    class Empty:
        def generate_rows(self, goal, count, banned):
            return "[]"

    materialize_dataset(plan, "x", Empty())
    assert plan.dataset.kind == "hf"
    assert plan.dataset.hf_id == "yahma/alpaca-cleaned"
    assert plan.dataset.note and "fell back" in plan.dataset.note
