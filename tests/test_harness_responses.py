"""###RESPONSES_JSON### marker extraction (spec 5.4)."""

import json

import pytest

from supertaco.eval.harness import RESPONSES_END, RESPONSES_START, split_responses


def _block(payload: dict) -> str:
    return f"{RESPONSES_START}\n{json.dumps(payload)}\n{RESPONSES_END}"


PAYLOAD = {"base": {"p": "a"}, "fine_tuned": {"p": "b"}}


def test_split_returns_train_logs_and_payload():
    logs = "step 4 loss 1.92\n" + _block(PAYLOAD) + "\ntraining finished"
    train_logs, parsed = split_responses(logs)
    assert parsed == PAYLOAD
    assert RESPONSES_START not in train_logs
    assert RESPONSES_END not in train_logs
    assert "step 4 loss 1.92" in train_logs
    assert "training finished" in train_logs


def test_no_block_returns_logs_unchanged():
    logs = "step 4 loss 1.92"
    assert split_responses(logs) == (logs, None)


def test_unterminated_block_returns_none_and_keeps_logs():
    logs = f'loss\n{RESPONSES_START}\n{{"base": {{}}'
    train_logs, parsed = split_responses(logs)
    assert parsed is None
    assert train_logs == logs


def test_bad_json_returns_none():
    logs = f"loss\n{RESPONSES_START}\nnot-json\n{RESPONSES_END}"
    train_logs, parsed = split_responses(logs)
    assert parsed is None
    assert RESPONSES_START not in train_logs  # block still stripped
    assert RESPONSES_END not in train_logs


def test_non_object_payload_returns_none():
    assert split_responses(f"{RESPONSES_START}\n[1, 2]\n{RESPONSES_END}")[1] is None


def test_non_string_values_rejected():
    block = _block({"base": {"p": 1}, "fine_tuned": {"p": "b"}})
    train_logs, parsed = split_responses(block)
    assert parsed is None
    assert RESPONSES_START not in train_logs  # block still stripped


def test_empty_payload_rejected():
    assert split_responses(f"{RESPONSES_START}\n{{}}\n{RESPONSES_END}")[1] is None


def test_deeply_nested_payload_returns_none():
    # 20000 exceeds this build's json recursion threshold (CPython 3.14.3:
    # 10000 parses, 15000 raises); raise the multiplier if that ever changes.
    inner = '{"a":' * 20000 + "1" + "}" * 20000
    with pytest.raises(RecursionError):  # fixture sanity: recursion really fires
        json.loads(inner)
    train_logs, parsed = split_responses(f"{RESPONSES_START}\n{inner}\n{RESPONSES_END}")
    assert parsed is None
    assert RESPONSES_START not in train_logs
    assert RESPONSES_END not in train_logs
