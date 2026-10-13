"""Config rendering + Colab transport (spec 5.2/5.3)."""

import ast
import py_compile
import re
from pathlib import Path

import pytest
import yaml

from supertaco.errors import ConfigurationError
from supertaco.gpu.transport import (
    CFG_END,
    CFG_START,
    DATA_END,
    DATA_START,
    MODEL_WHITELIST,
    ColabTransport,
    render_script,
    resolve_model,
)

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


def test_rendered_config_covers_all_subscripted_cfg_keys(tmp_path):
    # live-smoke regression: rendering with the default config (which carries
    # only the dashboard knobs) must not drop keys the on-VM script subscripts
    cfg = yaml.safe_load(
        (REPO / "configs" / "defaults" / "colab_t4.yaml").read_text(encoding="utf-8")
    )
    cfg["model"] = resolve_model(cfg["model"])
    out = render_script(cfg, ["p"], TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    block = text.split(CFG_START, 1)[1].split(CFG_END, 1)[0]
    rendered_cfg = ast.literal_eval(block.split("CFG = ", 1)[1].strip())
    subscripts = sorted({key for key in re.findall(r"CFG\[\s*['\"](\w+)['\"]\s*\]", text)})
    missing = [key for key in subscripts if key not in rendered_cfg]
    assert not missing, f"rendered script subscripts keys absent from rendered CFG: {missing}"


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


def test_template_contains_phase2_generation_and_block(tmp_path):
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "disable_adapter()" in text
    assert "###RESPONSES_JSON###" in text
    assert "###END_RESPONSES_JSON###" in text
    assert 'json.dumps({"base"' in text
    assert "max_new_tokens=150" in text
    assert "do_sample=False" in text
    assert "model.eval()" in text
    # responses block must land in exec-stream logs before the zip step (split_responses reads logs)
    assert text.index("###RESPONSES_JSON###") < text.index("shutil.make_archive")
    assert text.index("with model.disable_adapter():") < text.index("base_answers[p] = _gen(t)")
    assert '"base": base_answers' in text
    assert '"fine_tuned": ft_answers' in text


def test_template_training_arguments_honors_gradient_checkpointing(tmp_path):
    # playbook OOM patch flips gradient_checkpointing; the template must pass
    # it through to TrainingArguments or the patched config is a no-op
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "gradient_checkpointing=CFG.get" in text
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_checkpointing": True,
    }
    out = render_script(cfg, ["p"], TEMPLATE, tmp_path)
    rendered = out.read_text(encoding="utf-8")
    block = rendered.split(CFG_START, 1)[1].split(CFG_END, 1)[0]
    rendered_cfg = ast.literal_eval(block.split("CFG = ", 1)[1].strip())
    assert rendered_cfg["gradient_checkpointing"] is True
    py_compile.compile(str(out), doraise=True)


def test_template_has_data_region_and_dual_loader():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "# >>> SUPER_TACO_DATA >>>" in text
    assert "# <<< SUPER_TACO_DATA <<<" in text
    assert 'DATA.get("kind") == "hf"' in text
    assert "datasets.Dataset.from_list" in text
    assert "yahma/alpaca-cleaned" in text
    # region stays inside main(), before the eval-response block
    assert text.index("# >>> SUPER_TACO_DATA >>>") < text.index("###RESPONSES_JSON###")
    # _swap replaces count=1 — a duplicate marker would silently retarget the span
    assert text.count("# >>> SUPER_TACO_DATA >>>") == 1
    assert text.count("# <<< SUPER_TACO_DATA <<<") == 1
    assert text.index("# >>> SUPER_TACO_DATA >>>") < text.index('DATA.get("kind")')


def test_render_script_embeds_data_and_strips_it_from_cfg(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "dataset": {
            "kind": "synthetic",
            "rows": [{"instruction": "a", "input": "", "output": "b"}],
        },
    }
    out = render_script(cfg, ["p"], TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    data_block = text.split(DATA_START, 1)[1].split(DATA_END, 1)[0]
    rendered = ast.literal_eval(data_block.split("DATA = ", 1)[1].strip())
    assert rendered["kind"] == "synthetic"
    assert len(rendered["rows"]) == 1
    cfg_block = text.split(CFG_START, 1)[1].split(CFG_END, 1)[0]
    assert "dataset" not in cfg_block  # rows never double-embed into CFG
    assert "eval_prompts" not in cfg_block
    py_compile.compile(str(out), doraise=True)


def test_render_without_dataset_key_uses_alpaca_default(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4}
    out = render_script(cfg, ["p"], TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    data_block = text.split(DATA_START, 1)[1].split(DATA_END, 1)[0]
    rendered = ast.literal_eval(data_block.split("DATA = ", 1)[1].strip())
    assert rendered == {
        "kind": "hf",
        "hf_id": "yahma/alpaca-cleaned",
        "max_rows": 256,
        "rows": None,
    }


def test_render_synthetic_without_rows_falls_back_to_alpaca(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "dataset": {"kind": "synthetic", "max_rows": 100},
    }
    out = render_script(cfg, ["p"], TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    data_block = text.split(DATA_START, 1)[1].split(DATA_END, 1)[0]
    rendered = ast.literal_eval(data_block.split("DATA = ", 1)[1].strip())
    assert rendered["kind"] == "hf"
    assert rendered["hf_id"] == "yahma/alpaca-cleaned"
    assert rendered["max_rows"] == 100
    assert rendered.get("note")


def test_render_rows_without_kind_defaults_to_synthetic(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "dataset": {"rows": [{"instruction": "a", "input": "", "output": "b"}]},
    }
    out = render_script(cfg, ["p"], TEMPLATE, tmp_path)
    text = out.read_text(encoding="utf-8")
    data_block = text.split(DATA_START, 1)[1].split(DATA_END, 1)[0]
    rendered = ast.literal_eval(data_block.split("DATA = ", 1)[1].strip())
    assert rendered["kind"] == "synthetic"
    assert len(rendered["rows"]) == 1


def test_data_value_containing_markers_rejected(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "dataset": {"kind": "hf", "hf_id": "# >>> SUPER_TACO_CFG >>>"},
    }
    with pytest.raises(ValueError, match="template markers"):
        render_script(cfg, ["x"], TEMPLATE, tmp_path)


def test_non_mapping_dataset_rejected(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4, "dataset": ["oops"]}
    with pytest.raises(ValueError, match="dataset must be a mapping"):
        render_script(cfg, ["x"], TEMPLATE, tmp_path)


def test_prompt_containing_markers_rejected(tmp_path):
    cfg = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "learning_rate": 2e-4}
    with pytest.raises(ValueError, match="template markers"):
        render_script(cfg, ["# >>> SUPER_TACO_DATA >>>"], TEMPLATE, tmp_path)


def test_non_list_rows_rejected(tmp_path):
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "learning_rate": 2e-4,
        "dataset": {"kind": "synthetic", "rows": "notalist"},
    }
    with pytest.raises(ValueError, match="dataset rows must be"):
        render_script(cfg, ["x"], TEMPLATE, tmp_path)


def test_render_script_output_is_pure_ascii(tmp_path):
    """Non-ASCII in prompts/cfg/dataset must be folded before embedding.

    The colab CLI reads the script with the Windows locale codec (cp1252);
    multi-byte UTF-8 then mojibakes on the VM and breaks prompt-key matching.
    """
    cfg = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "learning_rate": 0.0005,
        "batch_size": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "num_epochs": 2,
        "note": "curly ’ quotes — folded",
        "dataset": {
            "kind": "synthetic",
            "rows": [
                {
                    "instruction": "What’s “fine”?",
                    "input": "",
                    "output": "A—B is fine.",
                }
            ],
        },
    }
    prompts = ["Draft a short clause for a non\u2011disclosure agreement."]
    out = render_script(cfg, prompts, TEMPLATE, tmp_path)
    text = out.read_bytes().decode("ascii")
    assert "non-disclosure" in text
    assert "curly ' quotes - folded" in text
    data_block = text.split(DATA_START, 1)[1].split(DATA_END, 1)[0]
    rendered = ast.literal_eval(data_block.split("DATA = ", 1)[1].strip())
    assert rendered["rows"][0]["instruction"] == 'What\'s "fine"?'
    assert rendered["rows"][0]["output"] == "A-B is fine."
