"""Generate broken configs for the 8 playbook failure modes.

Used by tests and demo to verify the agent's diagnosis and repair loop.
Each config is saved to configs/runs/ with a timestamp prefix.
"""

import yaml
import os
from datetime import datetime


def make_nan_loss_config() -> dict:
    """Config that causes NaN loss in first steps."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 1e-1,  # Too high - causes NaN
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
        "gradient_clip_norm": None,
        "data_format": "plain_text",
        "chat_template": None,
    }


def make_oom_config() -> dict:
    """Config that causes CUDA OOM."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 5e-5,
        "batch_size": 64,  # Too large - causes OOM
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
        "gradient_clip_norm": 1.0,
        "data_format": "plain_text",
        "chat_template": "llama-3",
    }


def make_loss_divergence_config() -> dict:
    """Config that causes loss divergence."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 1e-1,  # Too high - causes divergence
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 10,  # Too many epochs without warmup
        "gradient_clip_norm": None,  # No clipping - causes divergence
        "data_format": "plain_text",
        "chat_template": "llama-3",
    }


def make_loss_plateau_config() -> dict:
    """Config that causes loss plateau."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 1e-6,  # Too low - plateau
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 1,  # Too few epochs
        "gradient_clip_norm": 1.0,
        "data_format": "plain_text",
        "chat_template": "llama-3",
    }


def make_eval_regression_config() -> dict:
    """Config that causes eval regression."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 1e-4,
        "batch_size": 8,
        "lora_r": 64,  # Too high lora rank
        "lora_alpha": 32,
        "num_epochs": 10,  # Too many epochs overfits then regresses
        "gradient_clip_norm": 1.0,
        "data_format": "plain_text",
        "chat_template": "llama-3",
    }


def make_tokenizer_mismatch_config() -> dict:
    """Config that causes tokenizer mismatch."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
        "gradient_clip_norm": 1.0,
        "data_format": "plain_text",  # Wrong format for chat template
        "chat_template": "wrong-template",  # Invalid template
    }


def make_dataloader_stall_config() -> dict:
    """Config that causes dataloader stall."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 5e-5,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
        "gradient_clip_norm": 1.0,
        "num_workers": 20,  # Too many workers - causes stall
        "data_format": "plain_text",
        "chat_template": "llama-3",
    }


def make_gradient_explosion_config() -> dict:
    """Config that causes gradient explosion."""
    return {
        "model": "llama-3-8b",
        "learning_rate": 1e-1,  # Too high - explosive gradients
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 16,
        "num_epochs": 3,
        "gradient_clip_norm": None,  # No clipping - explosion
        "data_format": "plain_text",
        "chat_template": "llama-3",
    }


def write_config(config: dict, dest_dir: str = "configs/runs", mode_name: str = "unknown") -> str:
    """Write a config YAML file and return the path.

    Configs are named with timestamp to avoid overwriting.
    """
    os.makedirs(dest_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Use a slug based on the failure mode
    slug = "broken_" + list(config.keys())[0] if config else "config"
    filename = f"{timestamp}_{mode_name}_{slug}.yaml"
    filepath = os.path.join(dest_dir, filename)

    # YAML with 2-space indent, alphabetized keys
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=True, allow_unicode=True)

    return filepath


def generate_all_broken_configs(dest_dir: str = "configs/runs") -> list:
    """Generate all 8 broken configs and return their file paths."""
    configs = [
        ("NAN_LOSS", make_nan_loss_config),
        ("OOM", make_oom_config),
        ("LOSS_DIVERGENCE", make_loss_divergence_config),
        ("LOSS_PLATEAU", make_loss_plateau_config),
        ("EVAL_REGRESSION", make_eval_regression_config),
        ("TOKENIZER_MISMATCH", make_tokenizer_mismatch_config),
        ("DATALOADER_STALL", make_dataloader_stall_config),
        ("GRADIENT_EXPLOSION", make_gradient_explosion_config),
    ]

    paths = []
    for mode_name, make_func in configs:
        config = make_func()
        filepath = write_config(config, dest_dir, mode_name=mode_name)
        paths.append((mode_name, filepath))
        print(f"Generated {mode_name} config: {filepath}")

    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate broken configs for demo")
    parser.add_argument("--dir", default="configs/runs", help="Destination directory")
    args = parser.parse_args()
    generate_all_broken_configs(args.dir)