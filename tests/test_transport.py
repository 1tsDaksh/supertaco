"""Config rendering + Colab transport (spec 5.2/5.3)."""

import py_compile
from pathlib import Path

import pytest

from supertaco.errors import ConfigurationError
from supertaco.gpu.transport import MODEL_WHITELIST, ColabTransport, render_script, resolve_model

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "colab" / "train_lora.py"


def test_whitelist_keys_resolve_to_hf_ids():
    for key, hf in MODEL_WHITELIST.items():
        assert resolve_model(key) == hf
        assert resolve_model(key.upper()) == hf


def test_hf_id_passthrough_is_case_insensitive():
    assert resolve_model("qwen/qwen2.5-1.5b-instruct") == "Qwen/Qwen2.5-1.5B-Instruct"


def test_llama_3_8b_rejected_with_options():
    with pytest.raises(ConfigurationError, match="T4-viable"):
        resolve_model("llama-3-8b")


def test_missing_model_rejected():
    with pytest.raises(ConfigurationError, match="T4-viable"):
        resolve_model(None)
    with pytest.raises(ConfigurationError, match="T4-viable"):
        resolve_model("")


def test_render_script_embeds_config_and_prompts(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "learning_rate": 0.0005,
        "batch_size": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "num_epochs": 2,
    }
    prompts = ["p one", "p two"]
    out = render_script(cfg, prompts, TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert out.parent == tmp_path
    assert "'model': 'Qwen/Qwen2.5-1.5B-Instruct'" in text
    assert "'learning_rate': 0.0005" in text
    assert "'batch_size': 2" in text
    assert "EVAL_PROMPTS = ['p one', 'p two']" in text
    # template file itself untouched
    assert "SUPER_TACO_CFG" in TEMPLATE.read_text(encoding="utf-8")


def test_render_script_is_deterministic_per_config(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4}
    a = render_script(cfg, ["x"], TEMPLATE, tmp_path / "a").read_text(encoding="utf-8")
    b = render_script(cfg, ["x"], TEMPLATE, tmp_path / "b").read_text(encoding="utf-8")
    assert a == b


def test_rendered_script_is_valid_python(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4}
    out = render_script(cfg, ["prompt with 'quote' and\nnewline"], TEMPLATE, tmp_path)
    py_compile.compile(str(out), doraise=True)


def test_config_value_containing_markers_rejected(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "evil": "# >>> SUPER_TACO_PROMPTS >>>"}
    with pytest.raises(ValueError, match="template markers"):
        render_script(cfg, ["x"], TEMPLATE, tmp_path)


class _Outcome:
    def __init__(self, success=True, artifact=None, error=None, logs=None):
        self.success = success
        self.artifact = artifact
        self.error = error
        self.logs = logs or []


def test_transport_renders_and_calls_adapter(tmp_path, monkeypatch):
    captured = {}

    def fake_run(script, *, output_dir, on_line=None):
        captured["script"] = script
        captured["output_dir"] = output_dir
        if on_line:
            on_line("step 4 loss 1.9218")
        return _Outcome(success=True, artifact=tmp_path / "a.zip", logs=["step 4 loss 1.9218"])

    monkeypatch.setattr("supertaco.gpu.colab.run_training", fake_run)
    t = ColabTransport(script_template=TEMPLATE, output_dir=tmp_path / "out")
    lines = []
    res = t.run_attempt(
        {"model": "qwen2.5-0.5b", "learning_rate": 2e-4}, ["p1", "p2"], on_line=lines.append
    )
    assert lines == ["step 4 loss 1.9218"]
    assert res.error is None
    assert res.artifact == tmp_path / "a.zip"
    assert "step 4 loss 1.9218" in res.logs
    assert captured["output_dir"] == tmp_path / "out"
    rendered = captured["script"].read_text(encoding="utf-8")
    assert "'model': 'Qwen/Qwen2.5-0.5B-Instruct'" in rendered
    assert "EVAL_PROMPTS = ['p1', 'p2']" in rendered


def test_transport_maps_failed_outcome_to_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "supertaco.gpu.colab.run_training",
        lambda script, *, output_dir, on_line=None: _Outcome(
            success=False, error="download failed (training tail: boom)", logs=["boom"]
        ),
    )
    t = ColabTransport(script_template=TEMPLATE, output_dir=tmp_path / "out")
    res = t.run_attempt({"model": "qwen2.5-0.5b", "learning_rate": 2e-4}, ["p"])
    assert "download failed" in res.error
    assert res.artifact is None
