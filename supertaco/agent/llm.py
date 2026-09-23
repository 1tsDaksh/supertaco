import time
from supertaco.errors import ConfigurationError


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

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.call_log = []

    def _call(self, model_key: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Make an API call to Nemotron model via OpenAI-compatible endpoint."""
        model_id = self.MODELS[model_key]
        # In production, this would be an httpx call to the Token Factory
        # For now, simulate a response based on prompt content
        response_text = self._simulate_response(model_id, prompt)
        return response_text

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
        prompt = f"""Given the following training log, identify the failure mode from these possibilities:
NAN_LOSS, OOM, LOSS_DIVERGENCE, LOSS_PLATEAU, EVAL_REGRESSION, TOKENIZER_MISMATCH, DATALOADER_STALL, GRADIENT_EXPLOSION.

Log: {log_snippet}

Return only the failure mode key, or "UNKNOWN" if none match."""
        result = self._call("classify", prompt)
        return result.strip()

    def propose_patch(self, failure_key: str, config: dict, log_snippet: str) -> dict:
        """Propose a config patch for the given failure mode using the super model."""
        prompt = f"""Given a fine-tuning config and failure mode, propose a config delta to fix the issue.

Failure mode: {failure_key}
Current config: {config}
Training log excerpt: {log_snippet[:500]}

Return a JSON object with "fix" (string description) and "reason" (string) fields.
Only modify reasonable hyperparameters (learning_rate, batch_size, lora_r, lora_alpha, num_epochs, gradient_clip_norm, warmup_steps, num_workers, chat_template)."""

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
        self.call_log.append({
            "model": model_key,
            "tokens": tokens,
            "latency": latency,
            "escalation_reason": escalation_reason,
            "timestamp": time.time(),
        })