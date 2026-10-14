"""judge_job.py: parser parity with judges.py and end-to-end against a fake TF server."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from supertaco.eval.judges import JUDGE_PROMPT, _hash_score, _parse_score

_SCRIPT = Path(__file__).resolve().parents[1] / "supertaco" / "eval" / "judge_job.py"


def _load_job_module():
    spec = importlib.util.spec_from_file_location("judge_job", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


job = _load_job_module()


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("8", 8.0),
        ("Score: 8/10", 8.0),
        ("7 out of 10", 7.0),
        ("the scale is 0 to 10, final answer 6", 6.0),
        ("Score: -2", -2.0),
        ("no numbers here", None),
        ("15", 15.0),
    ],
)
def test_parse_score_matches_judges(reply, expected):
    assert job.parse_score(reply) == _parse_score(reply)
    assert (job.parse_score(reply) is None) == (expected is None)


def test_judge_prompt_identical_to_host():
    assert job.JUDGE_PROMPT == JUDGE_PROMPT


def test_hash_score_matches_host():
    assert job.hash_score("p", "r") == _hash_score("p", "r")


class _FakeTF:
    """OpenAI-compatible POST + presigned-PUT receiver on one port."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.post_paths = []
        self.put_body = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                outer.post_paths.append(self.path)
                idx = len(outer.post_paths) - 1
                content = outer.replies[idx] if idx < len(outer.replies) else "5"
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": content}}],
                        "usage": {"total_tokens": 11},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_PUT(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.put_body = self.rfile.read(length)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _responses_payload():
    prompts = [f"prompt {i}" for i in range(5)]
    return {
        "prompts": prompts,
        "base": {p: f"base answer {i}" for i, p in enumerate(prompts)},
        "fine_tuned": {p: f"ft answer {i}" for i, p in enumerate(prompts)},
    }


def _run_script(tmp_path, server, replies=None):
    responses_path = tmp_path / "responses.json"
    responses_path.write_text(json.dumps(_responses_payload()), encoding="utf-8")
    env = {
        **os.environ,
        "TOKEN_FACTORY_BASE_URL": f"http://127.0.0.1:{server.port}/v1",
        "TOKEN_FACTORY_API_KEY": "test-key",
        "SCORES_PUT_URL": f"http://127.0.0.1:{server.port}/scores",
        "RESPONSES_PATH": str(responses_path),
        "JUDGE_TIMEOUT_SECONDS": "5",
    }
    return subprocess.run(
        [sys.executable, str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=60
    )


def test_script_scores_all_and_puts_harness_shape(tmp_path):
    # 10 calls: prompt0 base, prompt0 ft, prompt1 base, ... -> replies in that order
    replies = ["8", "9", "7", "8", "6", "7", "9", "10", "5", "6"]
    server = _FakeTF(replies)
    try:
        proc = _run_script(tmp_path, server)
    finally:
        server.close()
    assert proc.returncode == 0, proc.stderr
    assert len(server.post_paths) == 10
    assert all(p.endswith("/chat/completions") for p in server.post_paths)
    payload = json.loads(server.put_body)
    assert payload["base_scores"] == [8.0, 7.0, 6.0, 9.0, 5.0]
    assert payload["fine_tuned_scores"] == [9.0, 8.0, 7.0, 10.0, 6.0]
    assert all(m.startswith("base:real / ft:real") for m in payload["modes"])
    assert len(payload["prompts"]) == 5
    assert len(payload["call_log"]) == 10


def test_script_retries_once_on_empty_then_scores(tmp_path):
    # per response: empty first, then a number (10 responses -> 20 calls)
    replies = []
    for i in range(10):
        replies += ["", "7"]
    server = _FakeTF(replies)
    try:
        proc = _run_script(tmp_path, server)
    finally:
        server.close()
    assert proc.returncode == 0, proc.stderr
    assert len(server.post_paths) == 20  # one retry per response
    payload = json.loads(server.put_body)
    assert all(s == 7.0 for s in payload["base_scores"] + payload["fine_tuned_scores"])


def test_script_unparseable_twice_falls_back_to_hash(tmp_path):
    replies = ["no numbers here"] * 20
    server = _FakeTF(replies)
    try:
        proc = _run_script(tmp_path, server)
    finally:
        server.close()
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(server.put_body)
    assert all("hash" in m for m in payload["modes"])
    for prompt, base, ft in zip(
        payload["prompts"], payload["base_scores"], payload["fine_tuned_scores"], strict=True
    ):
        assert base == _hash_score(prompt, f"base answer {payload['prompts'].index(prompt)}")


def test_script_missing_responses_exits_nonzero(tmp_path):
    server = _FakeTF([])
    try:
        env = {
            **os.environ,
            "TOKEN_FACTORY_BASE_URL": f"http://127.0.0.1:{server.port}/v1",
            "TOKEN_FACTORY_API_KEY": "k",
            "SCORES_PUT_URL": f"http://127.0.0.1:{server.port}/scores",
            "RESPONSES_PATH": str(tmp_path / "missing.json"),
        }
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=60
        )
    finally:
        server.close()
    assert proc.returncode != 0


def test_script_put_failure_exits_nonzero(tmp_path):
    server = _FakeTF(["8"] * 10)
    server.close()  # nothing listening -> PUT refused
    responses_path = tmp_path / "responses.json"
    responses_path.write_text(json.dumps(_responses_payload()), encoding="utf-8")
    env = {
        **os.environ,
        "TOKEN_FACTORY_BASE_URL": "http://127.0.0.1:1/v1",  # nothing there either
        "TOKEN_FACTORY_API_KEY": "k",
        "SCORES_PUT_URL": "http://127.0.0.1:1/scores",
        "RESPONSES_PATH": str(responses_path),
        "JUDGE_TIMEOUT_SECONDS": "2",
    }
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode != 0
