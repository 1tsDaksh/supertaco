import time
from pathlib import Path

from supertaco.paths import repo_root


def _logs_dir() -> Path:
    """Audit-log directory, anchored to the repo root (cwd-independent). Test seam."""
    return repo_root() / "logs"


class NemotronClient:
    """Nemotron model client with cost-routing escalation.

    Cost routing rule: start at nano; escalate to super on low confidence;
    ultra only when playbook misses. Log every escalation.
    """

    MODELS = {
        "classify": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        "patch": "nvidia/nemotron-3-super-120b-a12b",
        "deep_reason": "nvidia/Nemotron-3-Ultra-550b-a55b",
    }
    FAILURE_THRESHOLD = 2
    COOLDOWN_SECONDS = 60.0
    _NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 410, 422})

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.call_log: list[dict] = []
        self._consecutive_failures = 0
        self._circuit_open_until: float | None = None

    @property
    def _circuit_open(self) -> bool:
        """Breaker state; cooldown expiry closes it so the next call probes (half-open)."""
        if self._circuit_open_until is None:
            return False
        if time.monotonic() >= self._circuit_open_until:
            self._circuit_open_until = None
            return False
        return True

    @classmethod
    def _is_non_retryable(cls, exc: BaseException) -> bool:
        """Auth/client errors are permanent: one strike opens the breaker."""
        return getattr(exc, "status_code", None) in cls._NON_RETRYABLE_STATUS

    def _http_call(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple[str, int]:
        """Real Token Factory HTTP call. Returns (text, total_tokens). Test seam."""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    def _call(
        self,
        model_key: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        timeout: float | None = None,
        shape: str = "object",
    ) -> str:
        """Call Nemotron via Token Factory; on any failure fall back to heuristics.

        The breaker opens after FAILURE_THRESHOLD consecutive failures (or one
        non-retryable HTTP error) and closes again after COOLDOWN_SECONDS with
        a probe call, so a single transient timeout degrades only the calls
        around it instead of the rest of the client's life (spec D3).
        """
        model_id = self.MODELS[model_key]
        start = time.monotonic()
        if self._circuit_open:
            text = self._simulate_response(model_id, prompt, shape)
            self._record(
                model_key,
                model_id,
                text,
                mode="fallback",
                latency=0.0,
                tokens=0,
                error="circuit_open: earlier call failed",
            )
            return text
        try:
            text, tokens = self._http_call(
                model_id,
                prompt,
                max_tokens,
                temperature,
                self.timeout if timeout is None else timeout,
            )
        except Exception as exc:
            self._consecutive_failures += 1
            if self._is_non_retryable(exc) or self._consecutive_failures >= self.FAILURE_THRESHOLD:
                self._circuit_open_until = time.monotonic() + self.COOLDOWN_SECONDS
            text = self._simulate_response(model_id, prompt, shape)
            self._record(
                model_key,
                model_id,
                text,
                mode="fallback",
                latency=time.monotonic() - start,
                tokens=0,
                error=f"{type(exc).__name__}: {exc}",
            )
            return text
        self._consecutive_failures = 0
        self._circuit_open_until = None
        self._record(
            model_key,
            model_id,
            text,
            mode="real",
            latency=time.monotonic() - start,
            tokens=tokens,
            error=None,
        )
        return text

    def _record(
        self,
        model_key: str,
        model_id: str,
        text: str,
        *,
        mode: str,
        latency: float,
        tokens: int,
        error: str | None,
    ) -> None:
        """Append a call entry to the in-memory log and logs/llm_calls.jsonl.

        JSONL write failure is flagged, never raised (spec 6: warning only).
        """
        entry = {
            "model_key": model_key,
            "model_id": model_id,
            "mode": mode,
            "tokens": tokens,
            "latency": round(latency, 3),
            "error": error,
            "timestamp": time.time(),
            "response_preview": text[:200],
        }
        self.call_log.append(entry)
        try:
            import json

            log_dir = _logs_dir()
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "llm_calls.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            entry["jsonl_written"] = True
        except OSError as exc:
            import warnings

            entry["jsonl_written"] = False
            warnings.warn(f"llm_calls.jsonl write failed: {exc}", stacklevel=2)

    def _simulate_response(self, model_id: str, prompt: str, shape: str = "object") -> str:
        """Simulate Nemotron response based on model, prompt, and requested shape."""
        # Simple heuristic-based simulation for demo purposes
        if "classify" in model_id.lower() or "nano" in model_id.lower():
            # Nano: quick classification
            if "NaN" in prompt or "nan" in prompt:
                return "NAN_LOSS"
            elif "OOM" in prompt or "oom" in prompt:
                return "OOM"
            elif "divergence" in prompt.lower():
                return "LOSS_DIVERGENCE"
            elif "plateau" in prompt.lower():
                return "LOSS_PLATEAU"
            elif "regression" in prompt.lower():
                return "EVAL_REGRESSION"
            elif "tokenizer" in prompt.lower() or "template" in prompt.lower():
                return "TOKENIZER_MISMATCH"
            elif "stall" in prompt.lower():
                return "DATALOADER_STALL"
            elif "explosion" in prompt.lower() or "grad" in prompt.lower():
                return "GRADIENT_EXPLOSION"
            else:
                return "UNKNOWN"

        elif "super" in model_id.lower():
            # Super: patch/draft proposals; a rows call degrades to an empty batch
            if shape == "array":
                return "[]"
            return '{"fix": "lower_learning_rate", "reason": "model suggests LR reduction"}'

        else:
            # Ultra: deep reason for novel failures
            return '{"novel_error": true, "suggestion": "consult external docs"}'

    def classify_failure(self, log_snippet: str) -> str:
        """Classify the failure mode from a log snippet using the cheapest model."""
        prompt = f"""Given the following training log, identify the failure mode from \
these possibilities:
NAN_LOSS, OOM, LOSS_DIVERGENCE, LOSS_PLATEAU, EVAL_REGRESSION, TOKENIZER_MISMATCH, \
DATALOADER_STALL, GRADIENT_EXPLOSION.

Log: {log_snippet}

Return only the failure mode key, or "UNKNOWN" if none match."""
        result = self._call("classify", prompt)
        return result.strip()

    def propose_patch(self, failure_key: str, config: dict, log_snippet: str) -> dict:
        """Propose a config patch for the given failure mode using the super model."""
        prompt = f"""Given a fine-tuning config and failure mode, propose a config delta to \
fix the issue.

Failure mode: {failure_key}
Current config: {config}
Training log excerpt: {log_snippet[:500]}

Return a JSON object with "fix" (string description) and "reason" (string) fields.
Only modify reasonable hyperparameters (learning_rate, batch_size, lora_r, lora_alpha, num_epochs, \
gradient_clip_norm, warmup_steps, num_workers, chat_template)."""

        result = self._call("patch", prompt)
        # Try to parse JSON from response
        try:
            import json

            return json.loads(result)
        except (json.JSONDecodeError,):
            return {"fix": result.strip(), "reason": "parsed manually"}

    def deep_reason(self, failure_key: str, config: dict, log_snippet: str) -> dict:
        """Use the ultra model for novel/unknown failures."""
        prompt = f"""The playbook's 8 failure modes don't match this error. Deep reason.

Failure mode attempted: {failure_key}
Config: {config}
Log snippet: {log_snippet[:800]}

Return a JSON with "novel_error": true and "suggestion" (string) for what to try next.
May include Tavily doc lookup recommendations."""

        result = self._call("deep_reason", prompt)
        try:
            import json

            return json.loads(result)
        except (json.JSONDecodeError,):
            return {"novel_error": True, "suggestion": result.strip()}

    def plan_goal(self, goal: str) -> str:
        """Ask the super model to turn a free-text goal into a run-plan JSON object."""
        prompt = f"""You are a fine-tuning planner. Given the user's goal, return ONE JSON \
object with exactly these keys:
"model" (one of: qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, tinyllama-1.1b),
"learning_rate", "batch_size", "lora_r", "lora_alpha", "lora_dropout", "num_epochs",
"dataset" (object: {{"kind": "synthetic"}} when training rows should be written from the goal,
or {{"kind": "hf", "id": "<huggingface-dataset-id>", "rows": <int>}} when the goal names a \
public dataset),
"eval_prompts" (a JSON array of exactly 5 strings, each testing a skill the goal requires).

User goal: {goal}

Return ONLY the JSON object. No prose, no markdown."""
        return self._call("patch", prompt)

    def generate_rows(self, goal: str, count: int, banned: list[str]) -> str:
        """Generate alpaca-shaped training rows for a goal with the super model."""
        prompt = f"""Write exactly {count} supervised fine-tuning examples for the goal below.
Each example is a JSON object: {{"instruction": "...", "input": "", "output": "..."}}.
Rules:
- instruction: a realistic user request (1-2 sentences), <= 120 characters
- output: a high-quality answer in the style the goal implies, <= 400 characters
- diverse phrasings; never repeat an instruction from this list: {list(banned[:25])}

Goal: {goal}

Return ONLY a JSON array of objects. No prose, no markdown."""
        return self._call("patch", prompt, max_tokens=8192, timeout=180.0, shape="array")

    def log_call(self, model_key: str, tokens: int, latency: float, escalation_reason: str = None):
        """Log an LLM call for audit and cost tracking."""
        self.call_log.append(
            {
                "model": model_key,
                "tokens": tokens,
                "latency": latency,
                "escalation_reason": escalation_reason,
                "timestamp": time.time(),
            }
        )
