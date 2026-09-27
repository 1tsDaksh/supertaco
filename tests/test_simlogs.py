"""Config -> log generation: classification, healing, determinism, shapes."""

import math
from pathlib import Path

import pytest

from supertaco.agent.playbook import apply_default_fix, detect_failure
from supertaco.agent.simlogs import (
    classify_config_failure,
    extract_loss_points,
    generate_logs,
)

RUNS = Path(__file__).resolve().parents[1] / "configs" / "runs"

FIXTURE_CASES = [
    "NAN_LOSS",
    "OOM",
    "LOSS_DIVERGENCE",
    "LOSS_PLATEAU",
    "EVAL_REGRESSION",
    "TOKENIZER_MISMATCH",
    "DATALOADER_STALL",
    "GRADIENT_EXPLOSION",
]

# Sidebar demo configs, copied verbatim from ui/dashboard.py demo_options
DEMO_CASES = {
    "NAN_LOSS": {
        "learning_rate": 0.1,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
    },
    "OOM": {
        "learning_rate": 5e-5,
        "batch_size": 64,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
    },
    "LOSS_DIVERGENCE": {
        "learning_rate": 0.1,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 10,
    },
    "LOSS_PLATEAU": {
        "learning_rate": 1e-6,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 1,
    },
    "EVAL_REGRESSION": {
        "learning_rate": 1e-4,
        "batch_size": 8,
        "lora_r": 64,
        "lora_alpha": 32,
        "num_epochs": 10,
    },
}

HEALTHY = {
    "learning_rate": 2e-5,
    "batch_size": 8,
    "lora_r": 8,
    "lora_alpha": 16,
    "num_epochs": 3,
    "gradient_clip_norm": 1.0,
    "chat_template": "llama-3",
}


def _fixture(mode: str) -> dict:
    import yaml

    matches = [p for p in RUNS.glob("*.yaml") if mode in p.name and "_patched" not in p.name]
    assert matches, f"no fixture for {mode}"
    return yaml.safe_load(matches[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", FIXTURE_CASES)
def test_fixture_classifies_as_its_mode(mode):
    config = _fixture(mode)
    assert classify_config_failure(config) == mode
    assert detect_failure(generate_logs(config)) == mode


@pytest.mark.parametrize("mode", list(DEMO_CASES))
def test_sidebar_demo_classifies_as_its_mode(mode):
    config = DEMO_CASES[mode]
    assert classify_config_failure(config) == mode
    assert detect_failure(generate_logs(config)) == mode


@pytest.mark.parametrize("mode", FIXTURE_CASES)
def test_healing_property_fixture_then_fix_is_healthy(mode):
    config = _fixture(mode)
    healed = apply_default_fix(mode, config)
    assert classify_config_failure(healed) is None
    assert detect_failure(generate_logs(healed)) is None


def test_healthy_config_generates_no_failure():
    assert classify_config_failure(HEALTHY) is None
    assert detect_failure(generate_logs(HEALTHY)) is None


def test_generation_is_deterministic():
    config = _fixture("NAN_LOSS")
    assert generate_logs(config) == generate_logs(config)


def test_healthy_loss_series_decays_and_is_not_plateau():
    pts = extract_loss_points(generate_logs(HEALTHY))
    assert len(pts) == 30
    assert pts[0] > pts[-1]
    assert pts[-10] - pts[-1] > 0.5  # playbook plateau guard


def test_divergence_series_increases():
    pts = extract_loss_points(generate_logs(_fixture("LOSS_DIVERGENCE")))
    mid = len(pts) // 2
    assert sum(pts[mid:]) / len(pts[mid:]) > sum(pts[:mid]) / mid


def test_nan_series_contains_nan():
    pts = extract_loss_points(generate_logs(_fixture("NAN_LOSS")))
    assert any(math.isnan(v) for v in pts)


def test_precedence_multi_trigger_configs_surface_first_rule():
    # The rule order is load-bearing for multi-trigger healing chains: the
    # FIRST matching rule must win regardless of what else the config trips.
    assert classify_config_failure({"batch_size": 64, "learning_rate": 0.1}) == "OOM"
    assert classify_config_failure({"batch_size": 64, "num_workers": 20}) == "OOM"
    assert (
        classify_config_failure({"learning_rate": 0.1, "lora_r": 64, "num_epochs": 10})
        == "LOSS_DIVERGENCE"
    )
    assert (
        classify_config_failure(
            {
                "learning_rate": 1e-4,
                "lora_r": 64,
                "num_epochs": 10,
                "chat_template": "llama-3",
                "gradient_clip_norm": 1.0,
            }
        )
        == "EVAL_REGRESSION"
    )


@pytest.mark.parametrize(
    "config,mode",
    [
        ({"batch_size": 32, "learning_rate": 2e-5}, None),
        ({"batch_size": 33, "learning_rate": 2e-5}, "OOM"),
        ({"learning_rate": 0.04999}, None),
        ({"learning_rate": 0.05, "chat_template": None}, "NAN_LOSS"),
        ({"learning_rate": 1.5e-6}, None),
        ({"learning_rate": 1.4e-6}, "LOSS_PLATEAU"),
        ({"learning_rate": 2e-5, "lora_r": 31, "num_epochs": 10}, None),
        ({"learning_rate": 2e-5, "lora_r": 32, "num_epochs": 10}, "EVAL_REGRESSION"),
        ({"learning_rate": 0.05, "num_epochs": 8, "chat_template": "llama-3"}, "LOSS_DIVERGENCE"),
        (
            {
                "learning_rate": 0.05,
                "num_epochs": 8,
                "chat_template": "llama-3",
                "gradient_clip_norm": 1.0,
            },
            None,
        ),
    ],
)
def test_rule_table_boundaries(config, mode):
    assert classify_config_failure(config) == mode
