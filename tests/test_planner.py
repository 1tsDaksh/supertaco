"""Goal -> RunPlan: parse, G10 clamps, fallbacks, dataset materialization."""

import json

import pytest

from supertaco.data import generate_dataset  # noqa: F401  (import sanity: planner uses it)
from supertaco.eval.harness import DEFAULT_PROMPTS, PROMPT_SUITE_SIZE
from supertaco.planner import (
    DatasetSpec,  # noqa: F401  (import sanity: planner exports it)
    RunPlan,  # noqa: F401  (import sanity: planner exports it)
    default_plan,
    materialize_dataset,
    plan_run,
    plan_to_config,
    revise_plan,
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


def test_plan_run_ignores_revision_note_in_reply():
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, revision_note="sneaky"))))
    assert plan.plan_mode == "real"
    assert plan.revision_note is None


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


def test_plan_run_clamps_non_finite_floats():
    plan = plan_run(
        "x", _LLM(json.dumps(dict(GOOD, learning_rate=float("nan"), lora_dropout=float("inf"))))
    )
    assert plan.plan_mode == "real"
    assert plan.learning_rate == pytest.approx(2e-4)
    assert plan.lora_dropout == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("learning_rate", 1e-7, 1e-5),
        ("learning_rate", 9.0, 1e-3),
        ("batch_size", 0, 1),
        ("batch_size", 64, 4),  # clamp to 16, then model-size cap (1.5b) to 4
        ("lora_r", 1, 4),
        ("lora_r", 128, 32),
        ("lora_alpha", 1, 8),
        ("lora_alpha", 999, 64),
        ("num_epochs", 0, 1),
        ("num_epochs", 42, 3),
    ],
)
def test_plan_run_clamps_both_bounds(field, value, expected):
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, **{field: value}))))
    assert getattr(plan, field) == pytest.approx(expected)


def test_plan_run_clamps_dropout_bounds():
    low = plan_run("x", _LLM(json.dumps(dict(GOOD, lora_dropout=-1.0))))
    assert low.lora_dropout == 0.0
    high = plan_run("x", _LLM(json.dumps(dict(GOOD, lora_dropout=0.9))))
    assert high.lora_dropout == 0.3


def test_plan_run_clamps_hf_rows_edges():
    low = plan_run(
        "x", _LLM(json.dumps(dict(GOOD, dataset={"kind": "hf", "id": "squad", "rows": 3})))
    )
    high = plan_run(
        "x", _LLM(json.dumps(dict(GOOD, dataset={"kind": "hf", "id": "squad", "rows": 10000})))
    )
    assert low.dataset.max_rows == 50
    assert high.dataset.max_rows == 512


def test_plan_run_blank_prompt_strings_fall_back_to_defaults():
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, eval_prompts=[""] * 5))))
    assert plan.eval_prompts == DEFAULT_PROMPTS


def test_plan_run_unknown_dataset_kind_becomes_synthetic():
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, dataset={"kind": "csv"}))))
    assert plan.dataset.kind == "synthetic"


def test_plan_to_config_caps_batch_for_large_model():
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, model="qwen2.5-1.5b", batch_size=16))))
    assert plan.batch_size == 4  # joint model x batch ceiling, not just the 16 clamp
    assert plan_to_config(plan)["batch_size"] == 4


def test_plan_caps_batch_for_3b_model():
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, model="qwen2.5-3b", batch_size=16))))
    assert plan.batch_size == 2


def test_plan_keeps_batch_for_small_model():
    plan = plan_run("x", _LLM(json.dumps(dict(GOOD, model="qwen2.5-0.5b", batch_size=8))))
    assert plan.batch_size == 8  # cap >= requested batch: unchanged


class _ReviseLLM:
    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def revise_plan(self, goal, plan_json, feedback):
        self.calls.append((goal, plan_json, feedback))
        return self._reply


def test_revise_plan_updates_knobs_and_captures_note():
    plan = RunPlan(batch_size=4, num_epochs=1)
    llm = _ReviseLLM('{"batch_size": 2, "num_epochs": 2, "note": "lighter, shorter"}')
    revised = revise_plan("be more concise", plan, "pirate chef", llm)
    assert revised.batch_size == 2 and revised.num_epochs == 2
    assert revised.revision_note == "lighter, shorter"
    assert revised.plan_mode == "real"
    goal, plan_json, feedback = llm.calls[0]
    assert goal == "pirate chef" and feedback == "be more concise"
    assert '"batch_size": 4' in plan_json


def test_revise_plan_clamps_and_validates_like_plan_run():
    plan = RunPlan()
    llm = _ReviseLLM('{"batch_size": 999, "model": "not-a-model", "eval_prompts": ["x"]}')
    revised = revise_plan("f", plan, "g", llm)
    assert revised.batch_size == 16  # clamped to the T4 range; same as plan_run (alias skips cap)
    assert revised.model == "qwen2.5-0.5b"  # _validate_model fallback
    assert len(revised.eval_prompts) == PROMPT_SUITE_SIZE  # wrong count -> defaults

    capped = revise_plan("f", plan, "g", _ReviseLLM('{"batch_size": 999, "model": "qwen2.5-3b"}'))
    assert capped.model == "Qwen/Qwen2.5-3B-Instruct"
    assert capped.batch_size == 2  # T4 model x batch cap applied


def test_revise_plan_unparseable_keeps_previous_plan_flagged():
    plan = RunPlan(batch_size=2, revision_note=None)
    revised = revise_plan("f", plan, "g", _ReviseLLM("not json at all"))
    assert revised.batch_size == 2  # previous plan preserved
    assert revised.plan_mode == "fallback"
    assert "revision failed" in (revised.revision_note or "")
    assert "keeping the previous plan" in (revised.revision_note or "")


def test_revise_plan_without_dataset_key_keeps_existing_dataset():
    plan = RunPlan(
        dataset=DatasetSpec(
            kind="synthetic", rows=[{"instruction": "a", "input": "", "output": "b"}]
        )
    )
    revised = revise_plan("f", plan, "g", _ReviseLLM('{"num_epochs": 2}'))
    assert revised.dataset.kind == "synthetic"
    assert revised.dataset.rows == plan.dataset.rows
    assert revised.num_epochs == 2


def test_revise_plan_partial_reply_keeps_prior_knobs():
    plan = RunPlan(
        batch_size=4,
        learning_rate=5e-4,
        lora_r=16,
        num_epochs=2,
        eval_prompts=[f"prior {i}" for i in range(5)],
    )
    llm = _ReviseLLM('{"batch_size": 2, "note": "half the batch"}')
    revised = revise_plan("f", plan, "g", llm)
    assert revised.plan_mode == "real"
    assert revised.batch_size == 2
    assert revised.learning_rate == pytest.approx(5e-4)
    assert revised.lora_r == 16
    assert revised.num_epochs == 2
    assert revised.eval_prompts == plan.eval_prompts
    assert revised.revision_note == "half the batch"


def test_revise_plan_fallback_does_not_alias_prior_plan():
    plan = RunPlan(
        batch_size=2,
        dataset=DatasetSpec(
            kind="synthetic", rows=[{"instruction": "a", "input": "", "output": "b"}]
        ),
        eval_prompts=[f"prior {i}" for i in range(5)],
    )
    revised = revise_plan("f", plan, "g", _ReviseLLM("not json at all"))
    assert revised.plan_mode == "fallback"
    assert revised.batch_size == 2
    assert revised.dataset is not plan.dataset
    assert revised.dataset.rows is not plan.dataset.rows
    assert revised.eval_prompts is not plan.eval_prompts


def test_revise_plan_prompt_json_excludes_dataset_rows():
    rows = [{"instruction": f"row {n}", "input": "", "output": "o"} for n in range(40)]
    plan = RunPlan(dataset=DatasetSpec(kind="synthetic", rows=rows))
    llm = _ReviseLLM('{"num_epochs": 2}')
    revise_plan("f", plan, "g", llm)
    plan_json = llm.calls[0][1]
    cfg = json.loads(plan_json)
    assert cfg["dataset"] == {"kind": "synthetic", "max_rows": 256}  # rows stripped
    assert "row 0" not in plan_json  # training content never shipped to the model
    assert cfg["eval_prompts"] == plan.eval_prompts
