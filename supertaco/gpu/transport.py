"""Colab transport: render config -> script -> run on a fresh T4 VM."""

from __future__ import annotations

from supertaco.errors import ConfigurationError

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
