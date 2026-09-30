"""Config rendering + Colab transport (spec 5.2/5.3)."""

import pytest

from supertaco.errors import ConfigurationError
from supertaco.gpu.transport import MODEL_WHITELIST, resolve_model


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
