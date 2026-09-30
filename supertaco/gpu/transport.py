"""Colab transport: render config -> script -> run on a fresh T4 VM."""

from __future__ import annotations

import pprint
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Optional

from supertaco.errors import ConfigurationError
from supertaco.loop import AttemptResult

MODEL_WHITELIST: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def resolve_model(name: object) -> str:
    """Map a config model name to a T4-viable HF id or raise before provisioning."""
    if not name or not isinstance(name, str):
        raise ConfigurationError(
            f"model {name!r} is not T4-viable; pick from: {', '.join(sorted(MODEL_WHITELIST))}"
        )
    key = name.strip().lower()
    if key in MODEL_WHITELIST:
        return MODEL_WHITELIST[key]
    for hf_id in MODEL_WHITELIST.values():
        if key == hf_id.lower():
            return hf_id
    raise ConfigurationError(
        f"model {name!r} is not T4-viable; pick from: {', '.join(sorted(MODEL_WHITELIST))}"
    )


CFG_START = "# >>> SUPER_TACO_CFG >>>"
CFG_END = "# <<< SUPER_TACO_CFG <<<"
PROMPTS_START = "# >>> SUPER_TACO_PROMPTS >>>"
PROMPTS_END = "# <<< SUPER_TACO_PROMPTS <<<"


def _swap(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise ValueError(f"template markers missing: {start} .. {end}")
    return pattern.sub(lambda _m: replacement, text, count=1)


def render_script(config: dict, prompts: list[str], template: Path, out_dir: Path) -> Path:
    """Render a per-attempt training script with the config and prompts embedded."""
    text = template.read_text(encoding="utf-8")
    cfg_block = (
        CFG_START + "\n    CFG = " + pprint.pformat(dict(config), sort_dicts=True) + "\n" + CFG_END
    )
    prompts_block = (
        PROMPTS_START + "\n    EVAL_PROMPTS = " + repr(list(prompts)) + "\n" + PROMPTS_END
    )
    if any(m in cfg_block for m in (PROMPTS_START, PROMPTS_END)):
        raise ValueError("config values may not contain template markers")
    if any(m in prompts_block for m in (CFG_START, CFG_END)):
        raise ValueError("prompts may not contain template markers")
    text = _swap(text, CFG_START, CFG_END, cfg_block)
    text = _swap(text, PROMPTS_START, PROMPTS_END, prompts_block)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"attempt_{stamp}.py"
    path.write_text(text, encoding="utf-8")
    return path


class ColabTransport:
    """One attempt = fresh T4 VM: render -> train+generate -> download -> stop."""

    def __init__(self, script_template: Optional[Path] = None, output_dir: Optional[Path] = None):
        repo = Path(__file__).resolve().parents[2]
        self.script_template = script_template or repo / "colab" / "train_lora.py"
        self.output_dir = output_dir or repo / "colab" / "output"
        self.render_dir = repo / "configs" / "runs"

    def run_attempt(
        self, config: dict, prompts: list[str], on_line: Optional[Callable[[str], None]] = None
    ) -> AttemptResult:
        from supertaco.gpu.colab import run_training

        resolved = {**config, "model": resolve_model(config.get("model"))}
        script = render_script(resolved, prompts, self.script_template, self.render_dir)
        outcome = run_training(script, output_dir=self.output_dir, on_line=on_line)
        return AttemptResult(
            logs="\n".join(outcome.logs),
            artifact=outcome.artifact,
            error=outcome.error if not outcome.success else None,
        )
