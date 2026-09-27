import time


class NemotronClient:
    """Nemotron model client with cost-routing escalation.

    Cost routing rule: start at nano; escalate to super on low confidence;
    ultra only when playbook misses. Log every escalation.
    """

    MODELS = {
        "classify": "nvidia/nemotron-3-nano",
        "patch": "nvidia/nemotron-3-super",
        "deep_reason": "nvidia/nemotron-3-ultra",
    }

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.call_log: list[dict] = []
        self._circuit_open = False

    def _http_call(
        self, model_id: str, prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int]:
        """Real Token Factory HTTP call. Returns (text, total_tokens). Test seam."""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
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
        self, model_key: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> str:
        """Call Nemotron via Token Factory; on any failure fall back to heuristics.

        First failure opens the circuit for the rest of this client's life so a
        dead API costs one timeout instead of one per call (spec D3).
        """
        model_id = self.MODELS[model_key]
        start = time.monotonic()
        if self._circuit_open:
            text = self._simulate_response(model_id, prompt)
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
            text, tokens = self._http_call(model_id, prompt, max_tokens, temperature)
        except Exception as exc:
            self._circuit_open = True
            text = self._simulate_response(model_id, prompt)
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
            from pathlib import Path

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "llm_calls.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            entry["jsonl_written"] = True
        except OSError:
            entry["jsonl_written"] = False

    def _simulate_response(self, model_id: str, prompt: str) -> str:
        """Simulate Nemotron response based on model and prompt."""
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
            # Super: patch/draft proposals
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
