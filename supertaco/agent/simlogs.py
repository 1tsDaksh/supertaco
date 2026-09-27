"""Deterministic dry-run training-log generator.

Maps a training config to the log text a run would produce so the playbook
can classify dry-run output without a GPU. Rule precedence and healing
thresholds: docs/superpowers/specs/2026-09-24-functional-dashboard-design.md
section 5.1.
"""

from __future__ import annotations

import re
from typing import Optional

LOSS_RE = re.compile(r"\bloss\s+([0-9]+\.[0-9]+|NaN)\b")


def classify_config_failure(config: dict) -> Optional[str]:
    """Return the failure mode a config would trigger, or None if healthy.

    Missing keys are read as None/0 (num_epochs defaults to 3). Precedence
    order is significant: it is the order modes surface across relaunches
    for multi-trigger configs.
    """
    lr = float(config.get("learning_rate") or 0)
    batch = int(config.get("batch_size") or 0)
    workers = int(config.get("num_workers") or 0)
    epochs = int(config.get("num_epochs") or 3)
    lora_r = int(config.get("lora_r") or 0)
    clip = config.get("gradient_clip_norm")
    template = config.get("chat_template")

    if batch > 32:
        return "OOM"
    if workers > 16:
        return "DATALOADER_STALL"
    if template not in (None, "llama-3"):
        return "TOKENIZER_MISMATCH"
    if lr >= 0.05 and clip is None and epochs >= 8:
        return "LOSS_DIVERGENCE"
    if lora_r >= 32 and epochs >= 10:
        return "EVAL_REGRESSION"
    if lr >= 0.05 and clip is None and template == "llama-3":
        return "GRADIENT_EXPLOSION"
    if lr >= 0.05 and template in (None, ""):
        return "NAN_LOSS"
    if lr < 1.5e-6:
        return "LOSS_PLATEAU"
    return None


def _header(config: dict) -> str:
    return (
        "=== SuperTaco dry-run training (simulated Nebius job) ===\n"
        f"model={config.get('model', 'llama-3-8b')} "
        f"lr={config.get('learning_rate')} batch_size={config.get('batch_size')} "
        f"epochs={config.get('num_epochs')}\n"
    )


def _healthy_logs(config: dict) -> str:
    # Decay chosen so playbook's plateau window (last 10, must differ by >= 0.5)
    # and divergence check (second half not > 1.1x first) both stay negative.
    lines = [f"step {i} loss {2.4 - i * 0.06:.3f}" for i in range(30)]
    return _header(config) + "\n".join(lines) + "\ntraining finished (dry-run)\n"


def _nan_logs(config: dict) -> str:
    lines = [f"step {i} loss {2.4 - i * 0.05:.3f}" for i in range(12)]
    lines += ["step 13 loss NaN", "step 14 loss NaN", "FATAL: loss is NaN at step 14"]
    return _header(config) + "\n".join(lines) + "\n"


def _oom_logs(config: dict) -> str:
    lines = [f"step {i} loss {2.4 - i * 0.05:.3f}" for i in range(8)]
    lines += [
        "Traceback (most recent call last):",
        '  File "llamafactory/train/tuner.py", line 88, in train',
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    ]
    return _header(config) + "\n".join(lines) + "\n"


def _divergence_logs(config: dict) -> str:
    lines = [f"step {i} loss {1.0 + i * 0.18:.3f}" for i in range(30)]
    lines.append("WARN: loss trend rising over window")
    return _header(config) + "\n".join(lines) + "\n"


def _plateau_logs(config: dict) -> str:
    lines = [f"step {i} loss {2.0 + (0.001 if i % 2 else 0.0):.3f}" for i in range(25)]
    lines.append("WARN: no improvement in eval metric for 5 consecutive evals")
    return _header(config) + "\n".join(lines) + "\n"


def _grad_logs(config: dict) -> str:
    lines = [f"step {i} loss {1.8 - i * 0.01:.3f}" for i in range(10)]
    spikes = [8.42, 15.7, 42.11, 310.55]
    lines += [f"step {10 + i} lr 0.100 grad_norm {spikes[i % len(spikes)]}" for i in range(12)]
    lines.append("WARN: grad_norm spike detected, gradient explosion likely")
    return _header(config) + "\n".join(lines) + "\n"


def _eval_logs(config: dict) -> str:
    return (
        _header(config)
        + "\n".join(
            [
                "eval prompt_suite score 3.40 (baseline 6.10)",
                "post-train eval regression flagged vs baseline",
                "aborting: eval regression guard triggered",
            ]
        )
        + "\n"
    )


def _tokenizer_logs(config: dict) -> str:
    template = config.get("chat_template")
    return (
        _header(config)
        + "\n".join(
            [
                "starting dataset formatting pass",
                f"ValueError: chat template '{template}' not found in template registry",
                "tokenizer mismatch: dataset uses chat format but template is unknown",
                "aborting dataloader setup",
            ]
        )
        + "\n"
    )


def _stall_logs(config: dict) -> str:
    return (
        _header(config)
        + "\n".join(
            [
                "epoch 1/3 step 400/5000 metric unavailable (worker queue empty)",
                "no step progress in 15 minutes",
                "WARNING: dataloader workers idle; consider reducing num_workers",
                "no step progress in 16 minutes - stalling",
            ]
        )
        + "\n"
    )


_EMITTERS = {
    "NAN_LOSS": _nan_logs,
    "OOM": _oom_logs,
    "LOSS_DIVERGENCE": _divergence_logs,
    "LOSS_PLATEAU": _plateau_logs,
    "EVAL_REGRESSION": _eval_logs,
    "TOKENIZER_MISMATCH": _tokenizer_logs,
    "DATALOADER_STALL": _stall_logs,
    "GRADIENT_EXPLOSION": _grad_logs,
}


def generate_logs(config: dict) -> str:
    """Return the training log a run with this config would produce."""
    mode = classify_config_failure(config)
    if mode is None:
        return _healthy_logs(config)
    return _EMITTERS[mode](config)


def extract_loss_points(log_text: str) -> list[float]:
    """Extract loss values in emission order; NaN is preserved as float('nan')."""
    points: list[float] = []
    for match in LOSS_RE.finditer(log_text):
        raw = match.group(1)
        points.append(float("nan") if raw == "NaN" else float(raw))
    return points
