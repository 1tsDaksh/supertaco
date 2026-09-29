"""###RESPONSES_JSON### marker extraction (spec 5.4)."""

import json

from supertaco.eval.harness import RESPONSES_END, RESPONSES_START, split_responses


def _block(payload: dict) -> str:
    return f"{RESPONSES_START}\n{json.dumps(payload)}\n{RESPONSES_END}"


PAYLOAD = {"base": {"p": "a"}, "fine_tuned": {"p": "b"}}


def test_split_returns_train_logs_and_payload():
    logs = "step 4 loss 1.92\n" + _block(PAYLOAD) + "\ntraining finished"
    train_logs, parsed = split_responses(logs)
    assert parsed == PAYLOAD
    assert RESPONSES_START not in train_logs
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


def test_non_object_payload_returns_none():
    assert split_responses(f"{RESPONSES_START}\n[1, 2]\n{RESPONSES_END}")[1] is None


def test_payload_keys_must_be_strings():
    assert split_responses(f"{RESPONSES_START}\n{{}}\n{RESPONSES_END}")[1] is None
