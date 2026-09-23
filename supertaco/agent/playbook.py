from supertaco.errors import MaxRetriesExceeded

# Playbook v1 — 8 failure modes with typed detection signals and default fixes
PLAYBOOK = {
    "NAN_LOSS": {
        "detection": lambda log: any(
            "NaN" in line and "loss" in line.lower()
            for line in log.splitlines()[-20:]
        ),
        "default_fix": "lower_lr_10x",
        "description": "Loss is NaN in first N steps",
    },
    "OOM": {
        "detection": lambda log: "CUDA out of memory" in log or "out of memory" in log.lower(),
        "default_fix": "halve_batch_size",
        "description": "CUDA OOM in traceback",
    },
    "GRADIENT_EXPLOSION": {
        "detection": lambda log: any(
            "grad_norm" in line.lower()
            and len(line.split()) >= 2
            and line.split()[-1].replace(".", "").isdigit()
            and float(line.split()[-1]) > 1.0
            for line in log.splitlines()[-30:]
        ),
        "default_fix": "clip_grad_norm_lower_lr",
        "description": "Grad norm spikes",
    },
    "LOSS_DIVERGENCE": {
        "detection": lambda log: _loss_divergence_detected(log),
        "default_fix": "lower_lr_add_warmup_clip_grads",
        "description": "Loss increases > threshold over window",
    },
    "LOSS_PLATEAU": {
        "detection": lambda log: _plateau_detected(log),
        "default_fix": "raise_lr_slightly_or_adjust_schedule",
        "description": "No improvement over K evals",
    },
    "EVAL_REGRESSION": {
        "detection": lambda log: isinstance(log, dict) and log.get("regression", False),
        "default_fix": "lower_lora_rank_alpha_fewer_epochs_stronger_regularization",
        "description": "Post-train eval < baseline",
    },
    "TOKENIZER_MISMATCH": {
        "detection": lambda log: any(
            "template" in line.lower() or "tokenizer" in line.lower()
            for line in log.splitlines()[-30:]
        ),
        "default_fix": "fix_chat_template_verify_dataset_formatting",
        "description": "Template/tokenizer errors in logs",
    },
    "DATALOADER_STALL": {
        "detection": lambda log: "no step progress" in log.lower() and "minutes" in log.lower(),
        "default_fix": "reduce_num_workers_check_corrupted_shards",
        "description": "No step progress in T minutes",
    },
}


def _plateau_detected(log: str) -> bool:
    """Detect if loss has plateaued (no improvement over recent window)."""
    lines = log.splitlines()[-30:]
    loss_values = []
    for line in lines:
        stripped = line.strip()
        # Try to extract loss value from lines like "Step X loss: Y" or just "Y"
        for prefix in ["loss", "Loss", "Loss:"]:
            if prefix in stripped.lower():
                parts = stripped.split(prefix, 1)[-1].strip()
                try:
                    val = float(parts)
                    loss_values.append(val)
                except ValueError:
                    continue
                break
        # Also check for standalone numeric lines
        try:
            val = float(stripped)
            loss_values.append(val)
        except ValueError:
            continue
    if len(loss_values) < 3:
        return False
    # Check if last N values don't show significant improvement
    recent = loss_values[-10:]
    improvement = recent[0] - recent[-1]
    return abs(improvement) < 0.5


def _loss_divergence_detected(log: str) -> bool:
    """Detect if loss is diverging (increasing over a window)."""
    lines = log.splitlines()[-50:-1]
    loss_values = []
    for line in lines:
        stripped = line.strip()
        # Try to extract loss value from lines like "Step X loss: Y" or just "Y"
        for prefix in ["loss", "Loss", "Loss:"]:
            if prefix in stripped.lower():
                # Extract number after the prefix
                parts = stripped.split(prefix, 1)[-1].strip()
                try:
                    val = float(parts)
                    loss_values.append(val)
                except ValueError:
                    continue
                break
        # Also check for standalone numeric lines
        try:
            val = float(stripped)
            loss_values.append(val)
        except ValueError:
            continue
    if len(loss_values) < 3:
        return False
    # Check if values are generally increasing
    first_half_avg = sum(loss_values[: len(loss_values) // 2]) / (len(loss_values) // 2)
    second_half_avg = sum(loss_values[len(loss_values) // 2 :]) / (len(loss_values) - len(loss_values) // 2)
    return second_half_avg > first_half_avg * 1.1


def detect_failure(log: str) -> str | None:
    """Detect the failure mode from training log text. Returns the playbook key or None."""
    for key, spec in PLAYBOOK.items():
        if spec["detection"](log):
            return key
    return None


def apply_default_fix(failure_key: str, config: dict) -> dict:
    """Apply the default fix for a detected failure mode, returning a new config."""
    fixes = {
        "NAN_LOSS": lambda c: {**c, "learning_rate": c.get("learning_rate", 1e-4) / 10},
        "OOM": lambda c: {
            **c,
            "batch_size": max(1, c.get("batch_size", 4) // 2),
            "gradient_checkpointing": True,
        },
        "LOSS_DIVERGENCE": lambda c: {
            **c,
            "learning_rate": c.get("learning_rate", 1e-4) / 2,
            "gradient_clip_norm": 1.0,
            "warmup_steps": 10,
        },
        "LOSS_PLATEAU": lambda c: {
            **c,
            "learning_rate": c.get("learning_rate", 1e-4) * 1.5,
            "lora_r": max(8, c.get("lora_r", 8) // 2),
        },
        "EVAL_REGRESSION": lambda c: {
            **c,
            "lora_r": max(8, c.get("lora_r", 8) // 2),
            "lora_alpha": max(16, c.get("lora_alpha", 16) // 2),
            "num_epochs": max(1, c.get("num_epochs", 3) - 1),
        },
        "TOKENIZER_MISMATCH": lambda c: {
            **c,
            "chat_template": "llama-3",
            "dataset_format": "chat",
        },
        "DATALOADER_STALL": lambda c: {
            **c,
            "num_workers": max(1, c.get("num_workers", 4) // 2),
        },
        "GRADIENT_EXPLOSION": lambda c: {
            **c,
            "gradient_clip_norm": 0.5,
            "learning_rate": c.get("learning_rate", 1e-4) / 2,
        },
    }
    if failure_key not in fixes:
        raise ConfigurationError(f"No fix defined for failure mode: {failure_key}")
    return fixes[failure_key](config)