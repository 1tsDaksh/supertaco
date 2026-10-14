"""Nemotron judge inside a Nebius Serverless Job (stdlib only, no repo imports).

Reads responses.json, scores base vs fine-tuned via Token Factory
(OpenAI-compatible chat completions), PUTs scores.json to the presigned URL.
Mirrors supertaco/eval/judges.py semantics; tests/test_judge_job_script.py
enforces parity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

JUDGE_PROMPT = """You are a strict evaluation judge. Score the response from 0 to 10
(10 = excellent, 0 = unusable). Reply with a single number only.

Prompt: {prompt}
Response: {response}"""

JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"
MAX_TOKENS = 512
TEMPERATURE = 0.0


def parse_score(text):
    """Numerator of "8/10" wins; else the last number ("0 to 10" echoes lose)."""
    fraction = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:/|out of\s*)10\b", text)
    if fraction:
        return float(fraction.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else None


def hash_score(prompt, response):
    digest = int(hashlib.sha256(f"{prompt}{response}".encode()).hexdigest()[:8], 16)
    return round(2.0 + (digest % 81) / 10, 2)


def call_token_factory(base_url, api_key, prompt, timeout_s):
    """One OpenAI-compatible chat completion. Returns (text, mode, error, tokens, latency)."""
    body = json.dumps(
        {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        latency = time.monotonic() - start
        return "", "fallback", f"{type(exc).__name__}: {exc}", 0, latency
    text = ""
    try:
        text = payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        pass
    tokens = int((payload.get("usage") or {}).get("total_tokens") or 0)
    latency = time.monotonic() - start
    return text, "real", None, tokens, latency


def score_one(base_url, api_key, prompt, response, timeout_s):
    """Mirror of judges.score_response: retry once, then hash."""
    judge_prompt = JUDGE_PROMPT.format(prompt=prompt, response=response)
    text, http_mode, http_error, tokens, latency = "", "real", None, 0, 0.0
    for _ in range(2):
        text, http_mode, http_error, tokens, latency = call_token_factory(
            base_url, api_key, judge_prompt, timeout_s
        )
        raw = parse_score(text)
        if raw is not None:
            score = max(0.0, min(10.0, raw))
            return {
                "score": round(score, 2),
                "mode": http_mode,
                "error": http_error,
                "tokens": tokens,
                "latency": round(latency, 3),
                "response_preview": text[:200],
            }
    if http_mode == "fallback":
        mode = "fallback"
        error = (
            f"judge HTTP failure: {http_error or 'unknown transport error'}; "
            f"reply unparseable: {text[:60]!r}"
        )
    else:
        mode = "hash"
        error = f"unparseable judge output: {text[:80]!r}"
    return {
        "score": hash_score(prompt, response),
        "mode": mode,
        "error": error,
        "tokens": tokens,
        "latency": round(latency, 3),
        "response_preview": text[:200],
    }


def put_scores(put_url, payload, timeout_s):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        put_url, data=body, headers={"Content-Type": "application/json"}, method="PUT"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        resp.read()


def main():
    base_url = os.environ.get("TOKEN_FACTORY_BASE_URL", "")
    api_key = os.environ.get("TOKEN_FACTORY_API_KEY", "")
    put_url = os.environ.get("SCORES_PUT_URL", "")
    responses_path = os.environ.get("RESPONSES_PATH", "/supertaco/responses.json")
    timeout_s = float(os.environ.get("JUDGE_TIMEOUT_SECONDS", "120"))
    if not base_url or not api_key or not put_url:
        print(
            "missing TOKEN_FACTORY_BASE_URL / TOKEN_FACTORY_API_KEY / SCORES_PUT_URL",
            file=sys.stderr,
        )
        return 2
    try:
        with open(responses_path, encoding="utf-8") as fh:
            data = json.load(fh)
        prompts = data["prompts"]
        base = data["base"]
        ft = data["fine_tuned"]
        if not all(isinstance(base.get(p), str) and isinstance(ft.get(p), str) for p in prompts):
            raise ValueError("responses payload missing string answers")
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"bad responses payload: {exc}", file=sys.stderr)
        return 1
    results = {"base": [], "fine_tuned": []}
    modes, call_log = [], []
    for prompt in prompts:
        pair_modes = []
        for variant, answers in (("base", base), ("fine_tuned", ft)):
            res = score_one(base_url, api_key, prompt, answers[prompt], timeout_s)
            results[variant].append(res["score"])
            pair_modes.append(res["mode"])
            call_log.append(
                {
                    "prompt": prompt,
                    "variant": variant,
                    "mode": res["mode"],
                    "tokens": res["tokens"],
                    "latency": res["latency"],
                    "error": res["error"],
                    "response_preview": res["response_preview"],
                }
            )
        modes.append(f"base:{pair_modes[0]} / ft:{pair_modes[1]}")
    payload = {
        "prompts": prompts,
        "base_scores": results["base"],
        "fine_tuned_scores": results["fine_tuned"],
        "modes": modes,
        "call_log": call_log,
    }
    try:
        put_scores(put_url, payload, timeout_s)
    except (urllib.error.URLError, OSError) as exc:
        print(f"scores PUT failed: {exc}", file=sys.stderr)
        return 1
    print(f"scored {len(prompts)} prompts; PUT ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
