"""Playbook detection fixes required by the runner path."""

from pathlib import Path

from supertaco.agent.playbook import PLAYBOOK, apply_default_fix, detect_failure


def test_detects_eval_regression_from_log_text():
    log = "eval prompt_suite score 3.40\npost-train eval regression flagged vs baseline"
    assert detect_failure(log) == "EVAL_REGRESSION"


def test_eval_regression_does_not_match_unrelated_text():
    # Steep decay: a shallow series here would trip LOSS_PLATEAU, so the text
    # must avoid the plateau detector (last-10 window must differ by >= 0.5).
    assert detect_failure("step 1 loss 3.000\nstep 2 loss 2.000\nstep 3 loss 1.000") is None
    # Whole-log co-occurrence must not fire: the words on different lines, or
    # only one of them present, are not an eval regression.
    assert (
        detect_failure("training done\nno regression vs baseline detected\neval finished") is None
    )
    assert detect_failure("regression suite for v2 passed") is None
    assert detect_failure("eval_loss 1.234 at step 500") is None
    # And it must still fire case-insensitively.
    assert detect_failure("Eval Regression flagged vs baseline") == "EVAL_REGRESSION"


def test_apply_default_fix_unknown_key_raises_configuration_error():
    from supertaco.errors import ConfigurationError

    try:
        apply_default_fix("NOT_A_MODE", {"learning_rate": 1e-4})
    except ConfigurationError:
        pass
    else:
        raise AssertionError("ConfigurationError not raised")


def test_all_eight_modes_have_detection_and_fix():
    assert len(PLAYBOOK) == 8
    for key, spec in PLAYBOOK.items():
        assert callable(spec.get("detection"))
        apply_default_fix(
            key,
            {
                "learning_rate": 1e-4,
                "batch_size": 8,
                "lora_r": 8,
                "lora_alpha": 16,
                "num_epochs": 3,
            },
        )


def test_detects_lowercase_nan_from_torch_trainer():
    log = "step 4 loss nan\nstep 8 loss nan\nTraceback: loss became nan"
    assert detect_failure(log) == "NAN_LOSS"


def test_real_healthy_colab_log_detects_no_failure():
    fixture = Path(__file__).parent / "fixtures" / "real_colab_healthy.log"
    assert detect_failure(fixture.read_text(encoding="utf-8")) is None


def test_real_oom_traceback_detected():
    log = "step 4 loss 1.92\nCUDA out of memory. Tried to allocate 2.00 GiB"
    assert detect_failure(log) == "OOM"
