> **ABANDONED (2026-09-29):** Superseded by `docs/superpowers/plans/2026-09-29-colab-real-loop.md`.

# Real GPU Auto-Heal Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace dummy eval + flag-only regression behavior with a real Nebius GPU train+eval job, a real judge, and an automatic config-healing loop (regression feeds `EVAL_REGRESSION` back into the existing relaunch budget).

**Architecture:** One monolithic train+eval Serverless Job per attempt (Approach 1, spec D5). The real-jobs transport is a CLI subprocess (`nebius` CLI inside WSL2) — the only credential path proven to have jobs permission (spec §1, updated by Task 1). Job logs carry a delimited `###RESPONSES_JSON###` block; the runner extracts it and drives the existing classify→patch→relaunch loop with the judge's real scores.

**Tech Stack:** Python 3.11+, pydantic v2, Streamlit, pytest, ruff/mypy, `nebius` CLI 0.12.280 (WSL2 kali-linux), LLaMA-Factory image, Qwen2.5-1.5B-Instruct, L40S GPU.

**Spec:** `docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md` (commit `d5223b5`). Milestones 0–5 in order. GPU runs are approval-gated (AGENTS rule 6), marked **⏸ APPROVAL GATE**.

---

## Context the executor must know (zero-context briefing)

- **Shell:** Windows PowerShell 5.1. Use `;` not `&&`. `python -c "..."` is broken here — run scripts as files or `python -m ...`.
- **Gates (run every task):** `python -m pytest tests/ -m "not integration" -q`, `python -m ruff check .`, `python -m ruff format --check .`. mypy policy: no new errors on touched files (`python -m mypy supertaco/`).
- **Never commit:** `.env`, `logs/`, `supertaco/logs/`, `supertaco/configs/runs/*_patched.yaml`, `configs/runs/*_patched.yaml`. **Never modify:** `configs/base/` (empty — leave empty).
- **The Nebius CLI lives in WSL2.** Full invocation: `wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius`. Profile `default` configured (tenant `tenant-e00v26mtwqtan81xc4`, parent `project-e00w64x9pr00th10a6x8sv` = `default-project-eu-north1`). Non-elevated works. Do NOT put pipes/quotes/`$(...)` inside `wsl -- bash -c '...'` (PowerShell mangles them) — write script files under `\\wsl$\kali-linux\home\Zephyr\` instead.
- **CLI calls must be serial** — parallel invocations race the token cache (`~/.nebius/credentials.yaml`) and force spurious browser re-auths.
- **DNS hijack (Sky ISP):** `*.nebius.cloud` poisoned to `90.207.238.183`. Windows hosts + WSL `/etc/hosts` (with `generateHosts = false`) already fixed; preflight re-checks every run (spec §11). gRPC route hosts (`apps.msp.api.`, `cpl.iam.api.`, six others → `91.210.70.243`) are in WSL `/etc/hosts`.
- **Event catalog frozen at 10 types** (spec §4): `run_started, job_launched, logs_produced, failure_detected, classified, patch_proposed, patch_written, retry_scheduled, run_succeeded, run_failed`. Payload keys may grow; types may not.
- **Existing seams:** `log_fn`, `llm`, `runs_dir`, `on_event` on `runner.run`; `FakeLLM` + `_collect()` in `tests/test_runner.py`; `AppTest.from_file` + `offline` fixture in `tests/test_dashboard.py`.
- **PROJECT.md:** every task's commit step appends one line under `## Recent changes (newest first)` (heading `docs/PROJECT.md:26`), format: `- 2026-09-28: <area>: <change> (<plan Task N>).`
- **Design constraints from spec:** simulation stays default (D8); `--dry-run` always available, zero network (AGENTS rule 2); real mode never silently falls back to heuristics (D9); eval-regression relaunches share the initial+≤3 budget (D1); dry-run regression flag = config health (D6); event catalog unchanged; `RunResult` gains `eval_results`.

---

### Task 1: Milestone 0 — prerequisite verification + spec fact update

**Status: completed during the planning session (2026-09-28). This task records the verified facts.**

**Files:**
- Modify: `docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md` (§1 access facts, §8 milestone-0 row)
- Modify: `docs/PROJECT.md`

- [x] **Step 1: Verify jobs permission (already passed)**

Run: `nebius ai job list` via MCP tool → expected: `status: success`, empty output (HTTP 200). Direct equivalent: `wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius --no-check-update ai job list --format json` → `{}`.

- [x] **Step 2: Verify WSL hosts persistence**

Run: `wsl -d kali-linux -- grep -c nebius /etc/hosts` → expected ≥13 (5 base incl. `auth.nebius.com` + 8 gRPC route hosts). Run: `wsl -d kali-linux -- grep generateHosts /etc/wsl.conf` → `generateHosts = false`.

- [x] **Step 3: Replace spec §1 access-facts bullets**

Replace the two bullets under `### Verified access facts (probes run 2026-09-28)` with:

```markdown
- **Jobs access (verified 2026-09-28, milestone 0 complete):** the jobs-capable credential is
  the **user federation profile** (`nebius` CLI profile `default`, Google login, tenant
  `tenant-e00v26mtwqtan81xc4`, parent project `project-e00w64x9pr00th10a6x8sv` =
  `default-project-eu-north1`). `nebius ai job list` returns HTTP 200/empty — prerequisite P1
  satisfied. The `NEBIUS_API_KEY` AI Studio key (SA `sa-api-key-aiproject-…`, project
  `aiproject-e00b52k8gpcc7zs1me`) still has **no jobs rights** (403) and is NOT used for jobs;
  it remains the Token Factory/judge credential.
- **DNS hijack (verified + fixed 2026-09-28):** Sky broadband poisons `*.nebius.cloud` (and
  `kali.org`) to `90.207.238.183` (Sky block server — proven by apt errors referencing
  `block.isp.sky.com`). Fix is permanent: entries in Windows `hosts` AND WSL `/etc/hosts`
  (with `generateHosts = false`) covering `api.`, `api.eu-north1.`, `storage.eu-north1.`,
  `auth.nebius.com`, `apps.msp.api.`, `cpl.iam.api.` + six more route names (all →
  `91.210.70.243`). Prerequisite P2 satisfied; preflight (§5.1) re-checks every run.
- The **CLI-subprocess transport** (`wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius …`)
  is the chosen real-jobs transport (§5.1): only credential path with jobs permission, and its
  `ai job create/get/get-by-name/logs/cancel` verbs pin the request contract without guessing
  REST field names.
```

- [x] **Step 4: Update spec §8 milestone-0 row**

Change the milestone-0 row to:

```markdown
| 0 | **Prerequisites**: ✅ DONE 2026-09-28 — federation profile with jobs 200; Windows+WSL hosts entries (Sky hijack) | none |
```

- [x] **Step 5: Commit**

```powershell
git add docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md docs/PROJECT.md
git commit -m "docs: record milestone 0 access facts (federation jobs 200, Sky DNS hijack fix)"
```

Append to `docs/PROJECT.md` under `## Recent changes (newest first)`:

```markdown
- 2026-09-28: access: milestone 0 done — jobs list 200 via WSL CLI profile (federation, project-e00w64x9pr00th10a6x8sv); Sky DNS hijack (90.207.238.183) fixed in Windows+WSL hosts; spec §1/§8 updated (real-gpu-autoheal plan Task 1).
```

---

### Task 2: Milestone 1 — jobs API contract probe (live, free)

**Files:**
- Create: `docs/superpowers/specs/2026-09-28-jobs-contract-probe.md`

The spec (§5.1) requires the request contract be pinned by a live probe, never guessed. `nebius ai job create --dry-run` validates a request **without creating a resource** (free).

- [ ] **Step 1: Run the create dry-run probe**

```powershell
wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius --no-check-update ai job create --dry-run --format json --name supertaco-probe --image hiyouga/llamafactory:latest --container-command "bash -lc 'echo probe'" --platform gpu-l40s-d --preset 1gpu-16vcpu-200gb --timeout 1h --parent-id project-e00w64x9pr00th10a6x8sv
```

Expected: validation success. If the CLI rejects a flag *value* (image name, preset, timeout format), adjust values (not flags) until validation passes; record the accepted set.

- [ ] **Step 2: Capture read/cancel contracts**

```powershell
wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius ai job get --help
wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius ai job get-by-name --help
wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius ai job logs --help
wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius ai job cancel --help
```

Expected: `get --id`, `get-by-name --parent-id + --name` (both required), `logs --id --tail N [--follow|--since|--until]`, `cancel --id`. Note any deviation.

- [ ] **Step 3: Write the contract doc**

Create `docs/superpowers/specs/2026-09-28-jobs-contract-probe.md`:

```markdown
# Jobs API contract probe (milestone 1, plan Task 2) — 2026-09-28

Transport decision: **CLI subprocess** (only jobs-capable credential path; design spec §1).

## create (validated live via --dry-run on this date)
NebiusJobClient.launch_job builds: `ai job create` with flags
  --name <supertaco-YYYYmmddHHMMSS>  --image <builder image>
  --container-command "bash -lc 'llamafactory-cli train … && python …'"
  --platform gpu-l40s-d  --preset 1gpu-16vcpu-200gb  --timeout 1h
  --parent-id project-e00w64x9pr00th10a6x8sv  --working-dir /workspace
  --inject-file <local>:/workspace/<name> (x3)  --format json
Accepted value set observed: <fill from Step 1 output>

## read/cancel verbs
- get:         ai job get --id <id> --format json
- get-by-name: ai job get-by-name --parent-id <pid> --name <name> --format json
- logs:        ai job logs --id <id> --tail 100000      (plain-text stdout)
- cancel:      ai job cancel --id <id>

## response shapes
- create/get/get-by-name JSON: job id expected at .metadata.id (shape family confirmed by
  `iam v2 project list`, which returns .metadata.id).
- job state location/values: PINNED AT FIRST LIVE SMOKE (Task 8) — update
  NebiusJobClient._parse_job_state + its unit test to the observed shape.
```

- [ ] **Step 4: Gates + commit**

Run: `python -m pytest tests/ -m "not integration" -q` → all pass (no code touched); `python -m ruff check .` → clean.

```powershell
git add docs/superpowers/specs/2026-09-28-jobs-contract-probe.md docs/PROJECT.md
git commit -m "docs: pin jobs API contract via create --dry-run probe (milestone 1)"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: nebius: jobs API contract pinned by live create --dry-run probe; transport = WSL CLI subprocess (real-gpu-autoheal plan Task 2).
```

---

### Task 3: Recipe builder — JobSpec, LLaMA-Factory YAML, inject files, settings

**Files:**
- Create: `supertaco/nebius/recipe.py`
- Modify: `supertaco/settings.py` (add `nebius_cli` field + env extraction)
- Modify: `.env.example`
- Create: `tests/test_recipe.py`
- Modify: `docs/PROJECT.md`

The run config in this project is a small hyperparam dict (`learning_rate`, `batch_size`,
`lora_r`, `lora_alpha`, `num_epochs` — see `configs/runs/*.yaml`). The job needs a full
LLaMA-Factory train YAML. Recipe = fixed base LF template + hyperparam mapping + injected
files + the monolithic command chain (spec D5, D7).

- [ ] **Step 1: settings — CLI command + project id**

In `supertaco/settings.py`, add to `SandboxTuneSettings` (after `nebius_project_id`):

```python
    nebius_cli: str = Field(
        default="wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius",
        description="Command prefix used to invoke the jobs-capable nebius CLI",
    )
```

In `__init__`, extend the extraction list:

```python
        for field_name in ["nebius_api_key", "nebius_project_id", "tavily_api_key", "nebius_cli"]:
```

In `.env.example`, append:

```env
# Jobs-capable nebius CLI (command prefix, invoked via shlex.split)
NEBIUS_CLI=wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius
# Real jobs project (default-project-eu-north1); leave SuperTaco for simulation-only setups
# NEBIUS_PROJECT_ID=project-e00w64x9pr00th10a6x8sv
```

Local `.env` (NOT committed): set `NEBIUS_PROJECT_ID=project-e00w64x9pr00th10a6x8sv`.

- [ ] **Step 2: create `supertaco/nebius/recipe.py`**

```python
"""Recipe builder: run config -> Nebius Serverless Jobs request (spec 5.2, D5/D7)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import yaml

from supertaco.settings import settings

IMAGE = "hiyouga/llamafactory:latest"
PLATFORM = "gpu-l40s-d"
PRESET = "1gpu-16vcpu-200gb"
JOB_TIMEOUT = "1h"
WORKING_DIR = "/workspace"
MARKER_START = "###RESPONSES_JSON###"
MARKER_END = "###END_RESPONSES_JSON###"

# Full LLaMA-Factory SFT template for Qwen2.5-1.5B-Instruct + LoRA (spec D7).
BASE_LF_CONFIG: dict = {
    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "dataset": "alpaca_slice",
    "dataset_dir": f"{WORKING_DIR}/data",
    "template": "qwen",
    "cutoff_len": 1024,
    "max_steps": 60,
    "overwrite_cache": True,
    "preprocessing_num_workers": 4,
    "output_dir": f"{WORKING_DIR}/output",
    "logging_steps": 5,
    "save_strategy": "steps",
    "save_steps": 30,
    "save_total_limit": 1,
    "plot_loss": False,
    "overwrite_output_dir": True,
    "bf16": True,
}

# Run-config hyperparameter -> LLaMA-Factory field.
HYPERKEY_MAP = {
    "batch_size": "per_device_train_batch_size",
    "learning_rate": "learning_rate",
    "lora_r": "lora_rank",
    "lora_alpha": "lora_alpha",
    "num_epochs": "num_train_epochs",
}


@dataclass(frozen=True)
class JobSpec:
    name: str
    image: str
    platform: str
    preset: str
    timeout: str
    parent_id: str
    container_command: str
    working_dir: str = WORKING_DIR


@dataclass(frozen=True)
class PreparedJob:
    """A JobSpec plus files to inject, keyed by container path."""

    spec: JobSpec
    files: dict[str, str] = field(default_factory=dict)


def build_job_name(now: float | None = None) -> str:
    """Unique, get-by-name-able job name (spec 5.1 read path)."""
    return time.strftime("supertaco-%Y%m%d%H%M%S", time.localtime(now))


def build_llamafactory_yaml(config: dict) -> str:
    """Merge the run's hyperparams into the base LF template and render YAML."""
    lf = dict(BASE_LF_CONFIG)
    for src, dst in HYPERKEY_MAP.items():
        if config.get(src) is not None:
            lf[dst] = config[src]
    return yaml.safe_dump(lf, sort_keys=False, default_flow_style=False)


def build_job_spec(
    config: dict,
    *,
    prompts: list[str],
    prepare_script: str,
    eval_script: str,
    parent_id: str | None = None,
    job_name: str | None = None,
) -> PreparedJob:
    """Assemble the monolithic train+eval job (spec D5): 4 injected files, 1 command.

    prepare_script fetches the alpaca slice into /workspace/data; train runs
    llamafactory-cli; eval_script generates base vs fine-tuned responses; the
    command echoes the responses JSON between frozen markers (spec 4, D5).
    """
    parent = parent_id or settings.nebius_project_id
    if not parent:
        raise ValueError("No parent id: set NEBIUS_PROJECT_ID (default-project-eu-north1)")
    name = job_name or build_job_name()
    command = (
        "bash -lc 'set -euo pipefail && "
        "python /workspace/supertaco_prepare.py && "
        "llamafactory-cli train /workspace/train.yaml && "
        "python /workspace/supertaco_eval.py "
        "--prompts /workspace/prompts.json "
        "--responses /workspace/supertaco_responses.json && "
        f"echo {MARKER_START} && cat /workspace/supertaco_responses.json && echo && "
        f"echo {MARKER_END}'"
    )
    files = {
        f"{WORKING_DIR}/train.yaml": build_llamafactory_yaml(config),
        f"{WORKING_DIR}/prompts.json": json.dumps(prompts, indent=2),
        f"{WORKING_DIR}/supertaco_prepare.py": prepare_script,
        f"{WORKING_DIR}/supertaco_eval.py": eval_script,
    }
    return PreparedJob(
        spec=JobSpec(
            name=name,
            image=IMAGE,
            platform=PLATFORM,
            preset=PRESET,
            timeout=JOB_TIMEOUT,
            parent_id=parent,
            container_command=command,
        ),
        files=files,
    )
```

- [ ] **Step 3: create `tests/test_recipe.py`**

```python
"""Recipe builder: config -> LF YAML, job spec, injected files, markers."""

import json

import yaml

from supertaco.nebius.recipe import (
    MARKER_END,
    MARKER_START,
    build_job_name,
    build_job_spec,
    build_llamafactory_yaml,
)

PREP = "print('prepare')"
EVAL = "print('eval')"
PROMPTS = ["p1", "p2", "p3", "p4", "p5"]
CFG = {"learning_rate": 0.01, "batch_size": 8, "lora_r": 4, "lora_alpha": 8, "num_epochs": 2}


def _spec(**kw):
    return build_job_spec(
        CFG,
        prompts=PROMPTS,
        prepare_script=PREP,
        eval_script=EVAL,
        parent_id="project-test",
        job_name="supertaco-test",
        **kw,
    )


def test_build_job_name_has_prefix():
    assert build_job_name(0).startswith("supertaco-")


def test_llamafactory_yaml_merges_hyperparams():
    text = build_llamafactory_yaml(CFG)
    data = yaml.safe_load(text)
    assert data["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert data["dataset"] == "alpaca_slice"
    assert data["max_steps"] == 60
    assert data["learning_rate"] == 0.01
    assert data["per_device_train_batch_size"] == 8
    assert data["lora_rank"] == 4
    assert data["lora_alpha"] == 8
    assert data["num_train_epochs"] == 2
    assert data["bf16"] is True


def test_llamafactory_yaml_ignores_absent_keys():
    data = yaml.safe_load(build_llamafactory_yaml({"learning_rate": 0.1}))
    assert data["learning_rate"] == 0.1
    assert "lora_rank" not in data
    assert "num_train_epochs" not in data


def test_spec_fields_and_parent():
    job = _spec()
    assert job.spec.name == "supertaco-test"
    assert job.spec.image == "hiyouga/llamafactory:latest"
    assert job.spec.platform == "gpu-l40s-d"
    assert job.spec.preset == "1gpu-16vcpu-200gb"
    assert job.spec.timeout == "1h"
    assert job.spec.parent_id == "project-test"
    assert job.spec.working_dir == "/workspace"


def test_command_is_monolithic_with_markers():
    cmd = _spec().spec.container_command
    assert cmd.startswith("bash -lc '")
    assert "set -euo pipefail" in cmd
    assert "supertaco_prepare.py" in cmd
    assert "llamafactory-cli train /workspace/train.yaml" in cmd
    assert "supertaco_eval.py" in cmd
    assert MARKER_START in cmd and MARKER_END in cmd
    assert cmd.index("prepare") < cmd.index("llamafactory-cli")
    assert cmd.index(MARKER_START) < cmd.index(MARKER_END)


def test_files_cover_four_container_paths():
    job = _spec()
    assert set(job.files) == {
        "/workspace/train.yaml",
        "/workspace/prompts.json",
        "/workspace/supertaco_prepare.py",
        "/workspace/supertaco_eval.py",
    }
    assert job.files["/workspace/supertaco_prepare.py"] == PREP
    assert job.files["/workspace/supertaco_eval.py"] == EVAL
    assert json.loads(job.files["/workspace/prompts.json"]) == PROMPTS
    assert "learning_rate" in job.files["/workspace/train.yaml"]


def test_missing_parent_raises(monkeypatch):
    import pytest

    from supertaco.settings import settings

    monkeypatch.setattr(settings, "nebius_project_id", "")
    with pytest.raises(ValueError, match="NEBIUS_PROJECT_ID"):
        build_job_spec(CFG, prompts=PROMPTS, prepare_script=PREP, eval_script=EVAL)
```

- [ ] **Step 4: gates**

```powershell
python -m pytest tests/test_recipe.py -q          # expected: all passed
python -m pytest tests/ -m "not integration" -q   # expected: full suite green (76 + new)
python -m ruff check .                            # expected: Clean!
python -m ruff format --check .                   # expected: unchanged
```

- [ ] **Step 5: commit**

```powershell
git add supertaco/nebius/recipe.py supertaco/settings.py .env.example tests/test_recipe.py docs/PROJECT.md
git commit -m "feat: recipe builder — JobSpec, LF YAML merge, inject files, nebius_cli setting"
```

Append to `docs/PROJECT.md` under `## Recent changes (newest first)`:

```markdown
- 2026-09-28: nebius: recipe builder (JobSpec, LF YAML merge, 4 inject files, monolithic train+eval command) + settings.nebius_cli (real-gpu-autoheal plan Task 3).
```

---

### Task 4: Real `NebiusJobClient` — CLI-subprocess transport

**Files:**
- Modify: `supertaco/errors.py` (add `PreflightError`, `JobsApiError`)
- Rewrite: `supertaco/nebius/jobs.py`
- Create/extend: `tests/test_nebius_jobs.py`
- Modify: `docs/PROJECT.md`

Transport = the WSL `nebius` CLI (spec §5.1; the only credential path with jobs permission).
Flag spellings are the ones pinned by Task 2's `--help`/`--dry-run` probe — if the probe
differs, adjust this task's flags + tests together. If `--inject-file` does not exist,
fall back to base64-embedding the four files in the container command (same `PreparedJob`
interface; note it in the contract doc).

- [ ] **Step 1: errors**

Append to `supertaco/errors.py`:

```python
class PreflightError(Exception):
    """Raised when local preflight fails (DNS hijack, missing CLI) before launch."""


class JobsApiError(Exception):
    """Raised when the Nebius jobs CLI call fails."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr
```

- [ ] **Step 2: rewrite `supertaco/nebius/jobs.py`**

```python
"""Jobs client. Simulation: zero-network payload (dry_run). Real: WSL nebius CLI (spec 5.1)."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from supertaco.errors import JobsApiError, PreflightError
from supertaco.settings import settings

HIJACK_IP = "90.207.238.183"  # Sky block server poisoning *.nebius.cloud (spec 1)
TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

# Parallel CLI invocations race ~/.nebius/credentials.yaml -> forced re-auth.
_SERIAL_LOCK = threading.Lock()


@dataclass(frozen=True)
class JobStatus:
    id: str
    state: str
    message: str = ""


def _wsl_path(path: str) -> str:
    """Windows path -> /mnt/<drive>/... when the CLI runs inside WSL; else passthrough."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"
    return path


class NebiusJobClient:
    """Wrapper for Nebius Serverless Jobs (CLI-subprocess transport)."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        project_id: str = "",
        *,
        cli_cmd: Optional[str] = None,
        parent_id: Optional[str] = None,
        timeout_s: int = 120,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.project_id = project_id
        self.cli_cmd = cli_cmd if cli_cmd is not None else settings.nebius_cli
        self.parent_id = parent_id or project_id or settings.nebius_project_id
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ sim
    def launch_job(self, config: dict, dry_run: bool = False) -> dict:
        """Simulation payload (zero network). Unchanged contract for the dry-run path."""
        payload = {"projectId": self.project_id, "config": config}
        if dry_run:
            return {"dry_run": True, "payload": payload, "job_id": None}
        raise NotImplementedError("Use launch(prepared) for real jobs")

    # ---------------------------------------------------------------- real
    def _run(self, args: list[str], timeout_s: Optional[int] = None) -> subprocess.CompletedProcess:
        parts = shlex.split(self.cli_cmd)
        cmd = [*parts, "--no-check-update", *args]
        try:
            with _SERIAL_LOCK:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s or self.timeout_s,
                )
        except FileNotFoundError as exc:
            raise PreflightError(f"nebius CLI not runnable: {parts[0]!r} (set NEBIUS_CLI)") from exc
        except subprocess.TimeoutExpired as exc:
            raise JobsApiError(f"nebius CLI timed out: {' '.join(args[:4])}") from exc
        return proc

    def _ok(self, proc: subprocess.CompletedProcess, args: list[str]) -> str:
        if proc.returncode != 0:
            text = (proc.stderr or proc.stdout or "").strip()
            low = text.lower()
            if "unauthenticated" in low or "401" in low or "403" in low:
                text += "\nHint: jobs need the federation profile (spec 1); run `nebius profile list` in WSL."
            raise JobsApiError(
                f"nebius {' '.join(args[:4])} failed (exit {proc.returncode}): {text[-2000:]}",
                stderr=text,
            )
        return proc.stdout or ""

    def _json(self, proc: subprocess.CompletedProcess, args: list[str]) -> dict:
        out = self._ok(proc, args).strip()
        start = out.find("{")  # tolerate banner noise before the JSON
        if start < 0:
            raise JobsApiError(f"No JSON in `{' '.join(args[:4])}` output: {out[:300]}")
        try:
            data = json.loads(out[start:])
        except json.JSONDecodeError as exc:
            raise JobsApiError(f"Unparseable JSON from `{' '.join(args[:4])}`: {out[:300]}") from exc
        if not isinstance(data, dict):
            raise JobsApiError(f"Expected object JSON from `{' '.join(args[:4])}`")
        return data

    @staticmethod
    def _parse_job_state(data: dict, fallback_id: str = "") -> JobStatus:
        """Tolerant parse; exact shape pinned at first live smoke (Task 8, contract doc)."""
        meta = data.get("metadata") or {}
        job_id = data.get("id") or meta.get("id") or fallback_id
        raw = data.get("status")
        if isinstance(raw, dict):
            state = raw.get("state") or raw.get("phase") or "UNKNOWN"
            message = raw.get("message") or raw.get("reason") or ""
        elif isinstance(raw, str):
            state, message = raw, ""
        else:
            state = data.get("state") or "UNKNOWN"
            message = data.get("message") or ""
        return JobStatus(id=str(job_id), state=str(state).upper(), message=str(message))

    def preflight(self) -> None:
        """Block on missing CLI or poisoned DNS before spending anything (spec 11)."""
        parts = shlex.split(self.cli_cmd)
        if not shutil.which(parts[0]):
            raise PreflightError(f"nebius CLI not found: {parts[0]!r} (set NEBIUS_CLI)")
        try:
            infos = socket.getaddrinfo("api.nebius.cloud", 443, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise PreflightError(
                "Cannot resolve api.nebius.cloud — Sky DNS hijack? Re-apply the WSL hosts fix (spec 11)."
            ) from exc
        addrs = {info[4][0] for info in infos}
        if HIJACK_IP in addrs:
            raise PreflightError(
                f"DNS hijack: api.nebius.cloud -> {HIJACK_IP} (Sky block). "
                "Re-apply Windows+WSL hosts entries (spec 11)."
            )

    def launch(self, prepared) -> JobStatus:  # prepared: PreparedJob (recipe)
        spec = prepared.spec
        args = [
            "ai", "job", "create",
            "--name", spec.name,
            "--image", spec.image,
            "--container-command", spec.container_command,
            "--platform", spec.platform,
            "--preset", spec.preset,
            "--timeout", spec.timeout,
            "--parent-id", spec.parent_id,
            "--working-dir", spec.working_dir,
            "--format", "json",
        ]
        with tempfile.TemporaryDirectory(prefix="supertaco_job_") as tmp:
            for container_path, content in prepared.files.items():
                local = Path(tmp) / Path(container_path).name
                local.write_text(content, encoding="utf-8")
                args += ["--inject-file", f"{_wsl_path(str(local))}:{container_path}"]
            proc = self._run(args)
        data = self._json(proc, args)
        status = self._parse_job_state(data)
        if not status.id:
            raise JobsApiError(f"Create returned no job id: {json.dumps(data)[:300]}")
        return status

    def get_status(self, job_id: str) -> JobStatus:
        args = ["ai", "job", "get", "--id", job_id, "--format", "json"]
        return self._parse_job_state(self._json(self._run(args), args), fallback_id=job_id)

    def get_by_name(self, name: str, parent_id: Optional[str] = None) -> Optional[JobStatus]:
        args = [
            "ai", "job", "get-by-name",
            "--parent-id", parent_id or self.parent_id,
            "--name", name,
            "--format", "json",
        ]
        proc = self._run(args)
        if proc.returncode != 0:
            text = (proc.stderr or proc.stdout or "").lower()
            if any(t in text for t in ("not found", "notfound", "404", "does not exist")):
                return None
            self._ok(proc, args)
        return self._parse_job_state(self._json(proc, args))

    def get_logs(self, job_id: str, tail: int = 100000) -> str:
        args = ["ai", "job", "logs", "--id", job_id, "--tail", str(tail)]
        return self._ok(self._run(args), args)

    def stop_job(self, job_id: str) -> None:
        args = ["ai", "job", "cancel", "--id", job_id]
        proc = self._run(args)
        if proc.returncode != 0:
            text = (proc.stderr or proc.stdout or "").lower()
            if "not found" not in text:
                raise JobsApiError(f"cancel failed (exit {proc.returncode}): {text[-500:]}")

    def is_terminal(self, status: JobStatus) -> bool:
        return status.state in TERMINAL_STATES
```

- [ ] **Step 3: tests — `tests/test_nebius_jobs.py`**

```python
"""CLI-subprocess NebiusJobClient: arg building, parsing, preflight, errors."""

import subprocess
from pathlib import Path

import pytest

from supertaco.errors import JobsApiError, PreflightError
from supertaco.nebius.jobs import HIJACK_IP, TERMINAL_STATES, JobStatus, NebiusJobClient, _wsl_path
from supertaco.nebius.recipe import build_job_spec

PREP = "print('prepare')"
EVAL = "print('eval')"
PROMPTS = ["p1", "p2", "p3", "p4", "p5"]


def make_client(**kw):
    kw.setdefault("cli_cmd", "wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius")
    kw.setdefault("parent_id", "project-test")
    return NebiusJobClient(**kw)


def prepared(job_name="supertaco-test"):
    return build_job_spec(
        {"learning_rate": 0.01, "batch_size": 8},
        prompts=PROMPTS, prepare_script=PREP, eval_script=EVAL,
        parent_id="project-test", job_name=job_name,
    )


def cp(stdout="", stderr="", returncode=0, args=None):
    return subprocess.CompletedProcess(args or [], returncode, stdout=stdout, stderr=stderr)


def stub_run(client, responder):
    calls = []

    def fake(args, timeout_s=None):
        calls.append(list(args))
        return responder(args)

    client._run = fake
    return calls


def test_wsl_path_conversion():
    assert _wsl_path(r"C:\Users\x\AppData\Local\Temp\f") == "/mnt/c/Users/x/AppData/Local/Temp/f"
    assert _wsl_path("/home/z/f") == "/home/z/f"


def test_launch_builds_contract_args_and_parses_id():
    client = make_client()
    captured = {}

    def responder(args):
        captured["args"] = args
        for i, a in enumerate(args):
            if a == "--inject-file":
                local = args[i + 1].rsplit(":", 1)[0]
                assert Path(local).exists(), "inject file must exist during the call"
        return cp(stdout='{"metadata": {"id": "job-abc"}, "status": "Pending"}')

    stub_run(client, responder)
    status = client.launch(prepared())
    args = captured["args"]
    assert args[0:3] == ["ai", "job", "create"]
    for flag, value in [
        ("--name", "supertaco-test"),
        ("--image", "hiyouga/llamafactory:latest"),
        ("--platform", "gpu-l40s-d"),
        ("--preset", "1gpu-16vcpu-200gb"),
        ("--timeout", "1h"),
        ("--parent-id", "project-test"),
        ("--working-dir", "/workspace"),
        ("--format", "json"),
    ]:
        assert args[args.index(flag) + 1] == value
    injects = [args[i + 1] for i, a in enumerate(args) if a == "--inject-file"]
    assert len(injects) == 4
    assert all(p.startswith("/mnt/c/") for p in injects)
    assert any(p.endswith(":/workspace/train.yaml") for p in injects)
    cmd = args[args.index("--container-command") + 1]
    assert "###RESPONSES_JSON###" in cmd
    assert status == JobStatus(id="job-abc", state="PENDING", message="")


def test_launch_no_id_raises():
    client = make_client()
    stub_run(client, lambda args: cp(stdout='{"status": "Pending"}'))
    with pytest.raises(JobsApiError, match="no job id"):
        client.launch(prepared())


def test_nonzero_raises_with_profile_hint():
    client = make_client()
    stub_run(client, lambda args: cp(stderr="Unauthenticated", returncode=1))
    with pytest.raises(JobsApiError, match="federation profile"):
        client.get_status("j1")


def test_json_banner_noise_tolerated():
    client = make_client()
    stub_run(client, lambda args: cp(stdout='nebius cli banner\n{"metadata": {"id": "j2"}, "state": "running"}'))
    status = client.get_status("j2")
    assert status.id == "j2" and status.state == "RUNNING"


@pytest.mark.parametrize(
    "raw,expected_id,expected_state",
    [
        ({"metadata": {"id": "a"}, "status": {"state": "Running"}}, "a", "RUNNING"),
        ({"id": "b", "status": "SUCCEEDED"}, "b", "SUCCEEDED"),
        ({"state": "FAILED", "message": "oom"}, "c", "FAILED"),
    ],
)
def test_parse_job_state_shapes(raw, expected_id, expected_state):
    status = NebiusJobClient._parse_job_state(raw, fallback_id="c")
    assert status.id == expected_id
    assert status.state == expected_state


def test_get_logs_returns_plain_stdout():
    client = make_client()
    stub_run(client, lambda args: cp(stdout="step 1 loss\nstep 2 loss"))
    assert client.get_logs("j1") == "step 1 loss\nstep 2 loss"


def test_get_by_name_not_found_returns_none():
    client = make_client()
    stub_run(client, lambda args: cp(stderr="job not found", returncode=1))
    assert client.get_by_name("supertaco-x") is None


def test_get_by_name_found():
    client = make_client()
    stub_run(client, lambda args: cp(stdout='{"metadata": {"id": "j9"}, "status": "Running"}'))
    status = client.get_by_name("supertaco-x")
    assert status is not None and status.id == "j9"


def test_terminal_states():
    client = make_client()
    assert client.is_terminal(JobStatus("j", "SUCCEEDED"))
    assert client.is_terminal(JobStatus("j", "FAILED"))
    assert client.is_terminal(JobStatus("j", "CANCELLED"))
    assert not client.is_terminal(JobStatus("j", "RUNNING"))
    assert TERMINAL_STATES == {"SUCCEEDED", "FAILED", "CANCELLED"}


def test_preflight_blocks_hijacked_dns(monkeypatch):
    client = make_client()
    monkeypatch.setattr("supertaco.nebius.jobs.shutil.which", lambda p: "/usr/bin/wsl")
    monkeypatch.setattr(
        "supertaco.nebius.jobs.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", (HIJACK_IP, 443))],
    )
    with pytest.raises(PreflightError, match="hijack"):
        client.preflight()


def test_preflight_blocks_unresolvable_dns(monkeypatch):
    import socket as _socket

    client = make_client()
    monkeypatch.setattr("supertaco.nebius.jobs.shutil.which", lambda p: "/usr/bin/wsl")

    def boom(*a, **k):
        raise _socket.gaierror("NXDOMAIN")

    monkeypatch.setattr("supertaco.nebius.jobs.socket.getaddrinfo", boom)
    with pytest.raises(PreflightError, match="hijack"):
        client.preflight()


def test_preflight_blocks_missing_cli(monkeypatch):
    client = make_client()
    monkeypatch.setattr("supertaco.nebius.jobs.shutil.which", lambda p: None)
    with pytest.raises(PreflightError, match="NEBIUS_CLI"):
        client.preflight()


def test_preflight_passes_on_healthy_dns(monkeypatch):
    client = make_client()
    monkeypatch.setattr("supertaco.nebius.jobs.shutil.which", lambda p: "/usr/bin/wsl")
    monkeypatch.setattr(
        "supertaco.nebius.jobs.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("91.210.70.243", 443))],
    )
    client.preflight()  # no raise


def test_sim_launch_job_payload_unchanged():
    client = make_client(project_id="SuperTaco")
    out = client.launch_job({"learning_rate": 0.01}, dry_run=True)
    assert out == {
        "dry_run": True,
        "payload": {"projectId": "SuperTaco", "config": {"learning_rate": 0.01}},
        "job_id": None,
    }
```

- [ ] **Step 4: gates**

```powershell
python -m pytest tests/test_nebius_jobs.py tests/test_recipe.py -q   # expected: all passed
python -m pytest tests/ -m "not integration" -q                      # expected: green
python -m ruff check . ; python -m ruff format --check .             # expected: Clean! / unchanged
python -m mypy supertaco/nebius/jobs.py supertaco/nebius/recipe.py   # expected: no new errors
```

- [ ] **Step 5: commit**

```powershell
git add supertaco/errors.py supertaco/nebius/jobs.py tests/test_nebius_jobs.py docs/PROJECT.md
git commit -m "feat: real NebiusJobClient via WSL CLI subprocess — launch/get/logs/cancel + preflight"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: nebius: real jobs client (CLI subprocess: launch/get/get-by-name/logs/cancel, DNS+CLI preflight, JobsApiError/PreflightError) (real-gpu-autoheal plan Task 4).
```

---

### Task 5: Responses extraction — split marker block out of job logs

**Files:**
- Modify: `supertaco/eval/harness.py`
- Create: `tests/test_eval_signal.py`
- Modify: `docs/PROJECT.md`

Job logs carry the responses JSON between frozen markers (spec §4, D5). Two reasons to
split rather than just extract: (1) the success path needs the parsed responses; (2) the
prompt texts inside the block (e.g. *"How does a chat template differ from a
tokenizer?"*) would otherwise trip `detect_failure`'s TOKENIZER_MISMATCH rule on a
perfectly healthy run — the block must be stripped before playbook detection.

- [ ] **Step 1: add to `supertaco/eval/harness.py`**

```python
import json

RESPONSES_START = "###RESPONSES_JSON###"
RESPONSES_END = "###END_RESPONSES_JSON###"


def split_responses(log_text: str) -> tuple[str, dict | None]:
    """Split job logs into (train logs, responses block or None) (spec 4, D5).

    The marker block is stripped so prompt text inside it cannot masquerade as
    a training failure signal. Responses are None when markers are missing,
    the block is unterminated/unparseable, or the shape is not
    {"base": {prompt: str}, "fine_tuned": {prompt: str}} — fetch failure is
    not model failure (spec 5.5): callers skip eval, eval_results stays None.
    """
    start = log_text.find(RESPONSES_START)
    if start < 0:
        return log_text, None
    block_start = start + len(RESPONSES_START)
    end = log_text.find(RESPONSES_END, block_start)
    if end < 0:
        return log_text, None
    stripped = log_text[:start] + log_text[end + len(RESPONSES_END) :]
    blob = log_text[block_start:end].strip()
    try:
        data = json.loads(blob)
    except ValueError:
        return stripped, None
    if not isinstance(data, dict) or not {"base", "fine_tuned"} <= set(data):
        return stripped, None
    for key in ("base", "fine_tuned"):
        bucket = data[key]
        if not isinstance(bucket, dict) or not all(isinstance(v, str) for v in bucket.values()):
            return stripped, None
    return stripped, data


def extract_responses(log_text: str) -> dict | None:
    """Responses block only (thin wrapper over split_responses)."""
    return split_responses(log_text)[1]
```

- [ ] **Step 2: create `tests/test_eval_signal.py`**

```python
"""Marker extraction + train-log stripping (spec 4, D5, 5.5)."""

import json

from supertaco.agent.playbook import detect_failure
from supertaco.eval.harness import (
    RESPONSES_END,
    RESPONSES_START,
    extract_responses,
    split_responses,
)

GOOD = {"base": {"p1": "alpha"}, "fine_tuned": {"p1": "beta"}}
TEMPLATE_PROMPT = "How does a chat template differ from a tokenizer?"


def block(payload=GOOD) -> str:
    return f"{RESPONSES_START}\n{json.dumps(payload)}\n\n{RESPONSES_END}"


def test_split_separates_block():
    logs = f"loss 0.1\n{block()}\ntrain complete"
    train_logs, responses = split_responses(logs)
    assert responses == GOOD
    assert RESPONSES_START not in train_logs
    assert RESPONSES_END not in train_logs
    assert "loss 0.1" in train_logs and "train complete" in train_logs


def test_prompt_text_cannot_trigger_playbook():
    payload = {"base": {TEMPLATE_PROMPT: "x"}, "fine_tuned": {TEMPLATE_PROMPT: "y"}}
    train_logs, responses = split_responses(f"loss 0.1\n{block(payload)}")
    assert responses == payload
    assert detect_failure(train_logs) is None


def test_missing_markers_returns_original_logs():
    logs = "just training"
    assert split_responses(logs) == (logs, None)


def test_unterminated_block_returns_none_and_keeps_logs():
    logs = f"loss\n{RESPONSES_START}\n{{\"base\": {{}}"
    train_logs, responses = split_responses(logs)
    assert responses is None
    assert train_logs == logs


def test_invalid_json_returns_none_but_strips_block():
    logs = f"x\n{RESPONSES_START}\nnot-json\n{RESPONSES_END}"
    train_logs, responses = split_responses(logs)
    assert responses is None
    assert train_logs == "x\n"


def test_wrong_shape_returns_none():
    assert split_responses(f"{RESPONSES_START}\n[1, 2]\n{RESPONSES_END}")[1] is None
    missing = json.dumps({"base": {}, "other": 1})
    assert split_responses(f"{RESPONSES_START}\n{missing}\n{RESPONSES_END}")[1] is None
    not_dict = json.dumps({"base": {}, "fine_tuned": 5})
    assert split_responses(f"{RESPONSES_START}\n{not_dict}\n{RESPONSES_END}")[1] is None
    non_str = json.dumps({"base": {"p": 1}, "fine_tuned": {}})
    assert split_responses(f"{RESPONSES_START}\n{non_str}\n{RESPONSES_END}")[1] is None


def test_extract_responses_wrapper():
    assert extract_responses(f"a\n{block()}\nb") == GOOD
    assert extract_responses("nope") is None
```

- [ ] **Step 3: gates**

```powershell
python -m pytest tests/test_eval_signal.py -q     # expected: all passed
python -m pytest tests/ -m "not integration" -q   # expected: green
python -m ruff check . ; python -m ruff format --check .  # expected: Clean! / unchanged
```

- [ ] **Step 4: commit**

```powershell
git add supertaco/eval/harness.py tests/test_eval_signal.py docs/PROJECT.md
git commit -m "feat: split_responses — marker extraction with train-log stripping (spec 4/D5)"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: eval: split_responses/extract_responses (marker block stripped before playbook detection) (real-gpu-autoheal plan Task 5).
```

---

### Task 6: In-job scripts — prepare dataset + generate responses (spec D2/D5/D7)

**Files:**
- Create: `supertaco/nebius/jobscripts.py`
- Create: `scripts/gen_responses.py`
- Create: `tests/test_jobscripts.py`
- Modify: `docs/PROJECT.md`

`jobscripts.py` holds the two scripts injected into every real job (recipe Task 3 takes
them as `prepare_script`/`eval_script`). D2: real generation only — **no scoring inside
the job** (the judge runs locally via `eval_fn` in Task 7/12). D9: if the LoRA adapter was
not saved, the eval script exits nonzero so the job FAILS visibly instead of silently
scoring base-as-ft.

- [ ] **Step 1: create `supertaco/nebius/jobscripts.py`**

```python
"""Scripts injected into real Nebius jobs (recipe prepare_script/eval_script)."""

from __future__ import annotations

PREPARE_SCRIPT = '''
"""Fetch a small alpaca-cleaned slice into the LLaMA-Factory dataset layout."""
import json
import os

from datasets import load_dataset

OUT_DIR = "/workspace/data"
NAME = "alpaca_slice"
MAX_ROWS = 500


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset("yahma/alpaca-cleaned", split="train[:%d]" % MAX_ROWS)
    rows = [
        {
            "instruction": row["instruction"],
            "input": row["input"] or "",
            "output": row["output"],
        }
        for row in ds
    ]
    with open(os.path.join(OUT_DIR, NAME + ".json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    info = {
        NAME: {
            "file_name": NAME + ".json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    print("prepared %d rows at %s/%s.json" % (len(rows), OUT_DIR, NAME))


if __name__ == "__main__":
    main()
'''

EVAL_SCRIPT = '''
"""Generate base vs fine-tuned responses for the prompt suite (spec 5.3). Real only."""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "/workspace/output"
MAX_NEW_TOKENS = 160


def generate(model, tokenizer, prompts):
    out = {}
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out[prompt] = text.strip()
    return out


def load_adapter(base):
    names = os.listdir(ADAPTER_DIR) if os.path.isdir(ADAPTER_DIR) else []
    if not any(name.startswith("adapter_model") for name in names):
        raise SystemExit("no LoRA adapter saved under " + ADAPTER_DIR)
    return PeftModel.from_pretrained(base, ADAPTER_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--responses", required=True)
    args = parser.parse_args()
    with open(args.prompts, encoding="utf-8") as handle:
        prompts = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_responses = generate(base, tokenizer, prompts)
    del base
    torch.cuda.empty_cache()
    tuned = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tuned = load_adapter(tuned)
    tuned_responses = generate(tuned, tokenizer, prompts)
    with open(args.responses, "w", encoding="utf-8") as handle:
        json.dump(
            {"base": base_responses, "fine_tuned": tuned_responses},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print("responses written: %d prompts (base + fine_tuned)" % len(prompts))


if __name__ == "__main__":
    main()
'''
```

- [ ] **Step 2: create `scripts/gen_responses.py`**

Local helper: pull the responses block out of a saved job log (Task 8 smoke captures
logs to a file; demos reuse it offline).

```python
"""Extract the responses JSON block from a saved job log.

Usage: python scripts/gen_responses.py path/to/job.log [--out responses.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from supertaco.eval.harness import split_responses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    _, responses = split_responses(text)
    if responses is None:
        print("no valid responses block in " + str(args.log_file), file=sys.stderr)
        return 1
    blob = json.dumps(responses, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(blob, encoding="utf-8")
        print("wrote " + str(args.out))
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: create `tests/test_jobscripts.py`**

```python
"""Injected job scripts compile and contain the contract pieces (D2/D5/D7/D9)."""

import json
import subprocess
import sys

from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.nebius.jobscripts import EVAL_SCRIPT, PREPARE_SCRIPT


def test_prepare_script_compiles():
    compile(PREPARE_SCRIPT, "supertaco_prepare.py", "exec")


def test_eval_script_compiles():
    compile(EVAL_SCRIPT, "supertaco_eval.py", "exec")


def test_prepare_contract():
    assert "yahma/alpaca-cleaned" in PREPARE_SCRIPT
    assert "alpaca_slice" in PREPARE_SCRIPT
    assert "dataset_info.json" in PREPARE_SCRIPT
    assert "/workspace/data" in PREPARE_SCRIPT


def test_eval_contract_real_only_no_scoring():
    assert "peft" in EVAL_SCRIPT and "PeftModel" in EVAL_SCRIPT
    assert EVAL_SCRIPT.count('"base"') >= 1 and EVAL_SCRIPT.count('"fine_tuned"') >= 1
    # D2: no in-job scoring/regression verdicts — judge runs locally (eval_fn)
    assert "regression" not in EVAL_SCRIPT
    assert "score" not in EVAL_SCRIPT.lower()
    # D9: missing adapter must fail the job, not silently score base-as-ft
    assert "SystemExit" in EVAL_SCRIPT


def test_recipe_wires_both_scripts():
    from supertaco.nebius.recipe import build_job_spec

    job = build_job_spec(
        {"learning_rate": 0.01},
        prompts=["p"],
        prepare_script=PREPARE_SCRIPT,
        eval_script=EVAL_SCRIPT,
        parent_id="project-test",
        job_name="supertaco-test",
    )
    assert job.files["/workspace/supertaco_prepare.py"] == PREPARE_SCRIPT
    assert job.files["/workspace/supertaco_eval.py"] == EVAL_SCRIPT


def test_gen_responses_extracts_and_reports(tmp_path):
    payload = {"base": {"p1": "a"}, "fine_tuned": {"p1": "b"}}
    log = tmp_path / "job.log"
    log.write_text(
        "training...\n"
        + RESPONSES_START
        + "\n"
        + json.dumps(payload)
        + "\n"
        + RESPONSES_END
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "responses.json"
    proc = subprocess.run(
        [sys.executable, "scripts/gen_responses.py", str(log), "--out", str(out)],
        capture_output=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == payload

    empty = tmp_path / "empty.log"
    empty.write_text("no markers", encoding="utf-8")
    proc2 = subprocess.run(
        [sys.executable, "scripts/gen_responses.py", str(empty)],
        capture_output=True, encoding="utf-8",
    )
    assert proc2.returncode == 1
    assert "no valid responses block" in proc2.stderr
```

- [ ] **Step 4: gates**

```powershell
python -m pytest tests/test_jobscripts.py -q      # expected: all passed
python -m pytest tests/ -m "not integration" -q   # expected: green
python -m ruff check . ; python -m ruff format --check .  # expected: Clean! / unchanged
```

- [ ] **Step 5: commit**

```powershell
git add supertaco/nebius/jobscripts.py scripts/gen_responses.py tests/test_jobscripts.py docs/PROJECT.md
git commit -m "feat: in-job prepare+eval scripts (alpaca slice, real generation) + gen_responses helper"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: nebius: job scripts (prepare alpaca slice, real base/ft generation, D9 adapter check) + scripts/gen_responses.py (real-gpu-autoheal plan Task 6).
```

---

### Task 7: Runner real branch — launch, poll, heal loop, eval gate

**Files:**
- Modify: `supertaco/runner.py`
- Create: `tests/test_runner_real.py`
- Modify: `docs/PROJECT.md`

This replaces the `if not dry_run: ... not implemented yet (Gate 1)` stub. Simulation path
stays byte-for-byte unchanged (D8). Event catalog unchanged (payload keys may grow).
`eval_fn` is an injected seam (default wiring lands in Task 12); when its result reports
`regression_flagged`, the runner treats the attempt as an EVAL_REGRESSION failure and
continues the SAME heal loop with the shared relaunch budget (D1, spec 5.6).

- [ ] **Step 1: imports, docstring, RunResult**

`supertaco/runner.py` header becomes:

```python
"""Single run path shared by dashboard and CLI. Emits typed events.

Dry-run (default): Nebius launch builds a payload with zero network I/O.
Real mode: one monolithic train+eval Nebius job per attempt (spec 5); the
judge runs locally via the injected eval_fn seam. Never falls back (D9).
"""
```

Add `import time` and extend existing imports to:

```python
from supertaco.agent.playbook import apply_default_fix, detect_failure
from supertaco.agent.simlogs import extract_loss_points, generate_logs
from supertaco.errors import ConfigurationError, JobsApiError, PreflightError
from supertaco.eval.harness import DEFAULT_PROMPTS, split_responses
from supertaco.nebius.jobs import TERMINAL_STATES, NebiusJobClient
```

Extend `RunResult` (new field last — existing positional constructions keep working):

```python
@dataclass
class RunResult:
    success: bool
    attempts: int
    final_config: dict
    configs_written: list[str]
    failure_key: Optional[str]
    error: Optional[str]
    llm_calls: list[dict]
    eval_results: Optional[dict] = None
```

- [ ] **Step 2: `run()` signature + branch**

Add keyword-only params after `runs_dir`:

```python
    job_client: Any = None,
    poll_interval_s: float = 15.0,
    max_wait_s: float = 3600.0,
    eval_fn: Optional[Callable[[dict, list[str]], dict]] = None,
```

Replace the Gate-1 stub (current lines 99–102):

```python
    if not dry_run:
        return _run_real(
            config,
            max_retries=max_retries,
            on_event=on_event,
            llm=llm,
            runs_dir=runs_dir,
            job_client=job_client,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
            eval_fn=eval_fn,
        )
```

Everything after (sim path) unchanged. Update `run()` docstring: real mode returns
`run_failed` (never raises) on transport/preflight errors; `ConfigurationError` still
raises before any event.

- [ ] **Step 3: add `_run_real` + `_poll_until_done` (module level, after `run`)**

```python
def _poll_until_done(
    client: Any,
    job_id: str,
    attempt: int,
    on_event: Optional[OnEvent],
    poll_interval_s: float,
    max_wait_s: float,
) -> tuple[str, str, bool]:
    """Poll status+logs until terminal or deadline. Returns (logs, final_state, timed_out)."""
    deadline = time.monotonic() + max_wait_s
    last_text = ""
    while True:
        status = client.get_status(job_id)
        text = client.get_logs(job_id) or ""
        if text and text != last_text:
            last_text = text
            _emit(
                on_event, "logs_produced", attempt=attempt,
                text=text, loss_points=extract_loss_points(text),
            )
        if status.state in TERMINAL_STATES:
            return text, status.state, False
        if time.monotonic() >= deadline:
            try:
                client.stop_job(job_id)
            except JobsApiError:
                pass  # best-effort stop; the timeout itself is reported upstream
            return text, status.state, True
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)


def _run_real(
    config: dict,
    *,
    max_retries: int,
    on_event: Optional[OnEvent],
    llm: Any,
    runs_dir: str,
    job_client: Any,
    poll_interval_s: float,
    max_wait_s: float,
    eval_fn: Optional[Callable[[dict, list[str]], dict]],
) -> RunResult:
    """Real mode: monolithic train+eval job per attempt; heal via playbook (spec 5)."""
    from supertaco.nebius.jobscripts import EVAL_SCRIPT, PREPARE_SCRIPT
    from supertaco.nebius.recipe import build_job_spec

    client = job_client or NebiusJobClient()
    try:
        client.preflight()
    except PreflightError as exc:
        error = f"PreflightError: {exc}"
        _emit(on_event, "run_failed", error=error, attempts=0, failure_key=None, path=None)
        return RunResult(False, 0, dict(config), [], None, error, list(getattr(llm, "call_log", [])))

    current = dict(config)
    configs_written: list[str] = []
    attempt = 0
    attempt_limit = max_retries + 1
    failure_key: Optional[str] = None
    prompts = list(DEFAULT_PROMPTS)

    while attempt < attempt_limit:
        attempt += 1
        prepared = build_job_spec(
            current, prompts=prompts,
            prepare_script=PREPARE_SCRIPT, eval_script=EVAL_SCRIPT,
        )
        _emit(
            on_event, "job_launched", attempt=attempt,
            payload={
                "name": prepared.spec.name,
                "image": prepared.spec.image,
                "platform": prepared.spec.platform,
                "parent_id": prepared.spec.parent_id,
            },
        )
        try:
            launched = client.launch(prepared)
            log_text, final_state, timed_out = _poll_until_done(
                client, launched.id, attempt, on_event, poll_interval_s, max_wait_s
            )
        except (JobsApiError, PreflightError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            _emit(
                on_event, "run_failed", error=error, attempts=attempt,
                failure_key=failure_key,
                path=configs_written[-1] if configs_written else None,
            )
            return RunResult(
                False, attempt, current, configs_written, failure_key, error, list(getattr(llm, "call_log", []))
            )

        train_logs, responses = split_responses(log_text)
        failure_key = detect_failure(train_logs)
        eval_results: Optional[dict] = None
        eval_note: Optional[str] = None

        if failure_key is None and final_state == "SUCCEEDED" and not timed_out:
            if responses is None:
                eval_note = "responses JSON missing/unparseable — eval skipped (spec 5.5)"
            elif eval_fn is not None:
                eval_results = eval_fn(responses, prompts)
                if eval_results.get("regression_flagged"):
                    failure_key = "EVAL_REGRESSION"  # D1: heal on the shared budget
            if failure_key is None:
                _emit(
                    on_event, "run_succeeded", attempts=attempt,
                    configs_written=list(configs_written),
                    eval_results=eval_results, eval_note=eval_note,
                )
                return RunResult(
                    True, attempt, current, configs_written, None, None,
                    list(getattr(llm, "call_log", [])), eval_results=eval_results,
                )

        if failure_key is None:
            error = f"Job {launched.id} ended {final_state} without a playbook failure match"
            if timed_out:
                error += " (poll timeout — job stopped)"
            _emit(
                on_event, "run_failed", error=error, attempts=attempt,
                failure_key=None, path=configs_written[-1] if configs_written else None,
            )
            return RunResult(
                False, attempt, current, configs_written, None, error, list(getattr(llm, "call_log", []))
            )

        _emit(on_event, "failure_detected", attempt=attempt, failure_key=failure_key)

        call_log = getattr(llm, "call_log", [])
        verdict = llm.classify_failure(train_logs)
        last = call_log[-1] if call_log else {}
        _emit(
            on_event, "classified", attempt=attempt, failure_key=failure_key,
            nemotron_verdict=verdict, mode=last.get("mode", "unknown"),
            diverged=verdict != failure_key,
        )

        if attempt < attempt_limit:
            proposal = llm.propose_patch(failure_key, current, train_logs)
            _emit(
                on_event, "patch_proposed", attempt=attempt,
                failure_key=failure_key, proposal=proposal,
            )
            new_config = apply_default_fix(failure_key, current)
            path = _write_config(new_config, failure_key, Path(runs_dir))
            configs_written.append(path)
            _emit(
                on_event, "patch_written", attempt=attempt, path=path,
                failure_key=failure_key, before=current, after=new_config,
            )
            current = new_config
            _emit(on_event, "retry_scheduled", next_attempt=attempt + 1)

    error = f"MaxRetriesExceeded: {max_retries} relaunches exhausted"
    _emit(
        on_event, "run_failed", error=error, attempts=attempt,
        failure_key=failure_key, path=configs_written[-1] if configs_written else None,
    )
    return RunResult(
        False, attempt, current, configs_written, failure_key, error, list(getattr(llm, "call_log", []))
    )
```

- [ ] **Step 4: create `tests/test_runner_real.py`**

```python
"""Real-mode runner: launch, poll, heal loop, eval gate (spec 5, D1/D5/D9)."""

import json

import pytest

from supertaco.errors import ConfigurationError, JobsApiError, PreflightError
from supertaco.eval.harness import RESPONSES_END, RESPONSES_START
from supertaco.nebius.jobs import JobStatus
from supertaco.runner import run

FROZEN_EVENTS = {
    "run_started", "job_launched", "logs_produced", "failure_detected",
    "classified", "patch_proposed", "patch_written", "retry_scheduled",
    "run_succeeded", "run_failed",
}
CFG = {"learning_rate": 0.01, "batch_size": 8}
TEMPLATE_PROMPT = "How does a chat template differ from a tokenizer?"
TRICKY = {"base": {TEMPLATE_PROMPT: "x"}, "fine_tuned": {TEMPLATE_PROMPT: "y"}}
NAN_LOGS = "step 1 loss: 0.5\nstep 2: NaN loss detected"
OK_LOGS = "epoch 1 complete\nall steps finished"


@pytest.fixture(autouse=True)
def _project_id(monkeypatch):
    from supertaco.settings import settings

    monkeypatch.setattr(settings, "nebius_project_id", "project-test")


class FakeLLM:
    def __init__(self):
        self.call_log = []

    def classify_failure(self, logs):
        self.call_log.append({"kind": "classify"})
        return "NAN_LOSS"

    def propose_patch(self, failure_key, config, logs):
        self.call_log.append({"kind": "patch"})
        return "lower the learning rate"


class FakeJobClient:
    """Injected transport seam (spec 6): one scripted entry per launch."""

    def __init__(self, attempts, preflight_error=None):
        self.attempts = list(attempts)
        self.preflight_error = preflight_error
        self.launched = []
        self.stopped = []
        self._idx = -1
        self._logs = ""

    def preflight(self):
        if self.preflight_error is not None:
            raise self.preflight_error

    def launch(self, prepared):
        self._idx += 1
        self.launched.append(prepared.spec.name)
        self._logs = self.attempts[self._idx]["logs"]
        return JobStatus(id=f"job-{self._idx}", state="PENDING")

    def get_status(self, job_id):
        return JobStatus(id=job_id, state=self.attempts[self._idx]["state"])

    def get_logs(self, job_id):
        return self._logs

    def stop_job(self, job_id):
        self.stopped.append(job_id)


def responses_block(payload):
    return RESPONSES_START + "\n" + json.dumps(payload) + "\n" + RESPONSES_END


def eval_ok(responses, prompts):
    assert set(responses) == {"base", "fine_tuned"}
    assert len(prompts) == 5
    return {"regression_flagged": False, "base_avg": 0.7, "ft_avg": 0.9}


def do_run(tmp_path, client, **kw):
    events = []
    kw.setdefault("llm", FakeLLM())
    result = run(
        CFG, dry_run=False, on_event=events.append, runs_dir=str(tmp_path),
        job_client=client, poll_interval_s=0.0, max_wait_s=2.0, **kw,
    )
    return result, events


def types(events):
    return [e.type for e in events]


def test_happy_path_success_with_eval(tmp_path):
    client = FakeJobClient(
        [{"state": "SUCCEEDED", "logs": OK_LOGS + "\n" + responses_block(TRICKY)}]
    )
    result, events = do_run(tmp_path, client, eval_fn=eval_ok)
    assert result.success is True and result.attempts == 1
    assert result.configs_written == []
    assert result.eval_results == {"regression_flagged": False, "base_avg": 0.7, "ft_avg": 0.9}
    seen = types(events)
    assert seen[0] == "run_started" and seen[-1] == "run_succeeded"
    launch = next(e for e in events if e.type == "job_launched")
    assert launch.data["payload"]["name"].startswith("supertaco-")
    assert launch.data["payload"]["image"] == "hiyouga/llamafactory:latest"
    assert launch.data["payload"]["parent_id"] == "project-test"
    logs_ev = next(e for e in events if e.type == "logs_produced")
    assert "loss_points" in logs_ev.data
    assert set(seen) <= FROZEN_EVENTS


def test_success_without_responses_skips_eval(tmp_path):
    client = FakeJobClient([{"state": "SUCCEEDED", "logs": OK_LOGS}])
    called = []

    def eval_fn(responses, prompts):
        called.append(True)
        return {}

    result, events = do_run(tmp_path, client, eval_fn=eval_fn)
    assert result.success is True and result.eval_results is None
    assert called == []
    ev = next(e for e in events if e.type == "run_succeeded")
    assert "eval skipped" in ev.data["eval_note"]


def test_eval_regression_feeds_heal_loop(tmp_path):
    attempts = [
        {"state": "SUCCEEDED", "logs": OK_LOGS + "\n" + responses_block(TRICKY)},
        {"state": "SUCCEEDED", "logs": OK_LOGS + "\n" + responses_block(TRICKY)},
    ]
    client = FakeJobClient(attempts)
    calls = []

    def eval_gate(responses, prompts):
        calls.append(len(calls))
        return {"regression_flagged": len(calls) == 1}

    result, events = do_run(tmp_path, client, eval_fn=eval_gate)
    assert result.success is True and result.attempts == 2
    assert len(result.configs_written) == 1
    seen = types(events)
    assert "failure_detected" in seen and "patch_written" in seen
    fd = next(e for e in events if e.type == "failure_detected")
    assert fd.data["failure_key"] == "EVAL_REGRESSION"
    assert result.final_config["num_epochs"] == 2  # EVAL_REGRESSION default fix
    assert set(seen) <= FROZEN_EVENTS


def test_failed_job_classified_and_relaunched(tmp_path):
    client = FakeJobClient(
        [
            {"state": "FAILED", "logs": NAN_LOGS},
            {"state": "SUCCEEDED", "logs": OK_LOGS + "\n" + responses_block(TRICKY)},
        ]
    )
    result, events = do_run(tmp_path, client, eval_fn=eval_ok)
    assert result.success is True and result.attempts == 2
    assert len(result.configs_written) == 1
    assert result.final_config["learning_rate"] == pytest.approx(0.001)
    classified = next(e for e in events if e.type == "classified")
    assert classified.data["failure_key"] == "NAN_LOSS"
    assert classified.data["diverged"] is False
    assert result.eval_results["regression_flagged"] is False


def test_retries_exhausted(tmp_path):
    client = FakeJobClient([{"state": "FAILED", "logs": NAN_LOGS}] * 4)
    result, events = do_run(tmp_path, client)
    assert result.success is False
    assert result.attempts == 4 and len(result.configs_written) == 3
    assert "MaxRetriesExceeded" in result.error
    assert types(events)[-1] == "run_failed"


def test_launch_error_becomes_run_failed(tmp_path):
    class Boom(FakeJobClient):
        def launch(self, prepared):
            raise JobsApiError("Unauthenticated", stderr="Unauthenticated")

    result, events = do_run(tmp_path, Boom([]))
    assert result.success is False and result.attempts == 1
    assert "JobsApiError" in result.error
    assert types(events)[-1] == "run_failed"


def test_preflight_error_fails_before_launch(tmp_path):
    client = FakeJobClient([], preflight_error=PreflightError("DNS hijack: 90.207.238.183"))
    result, events = do_run(tmp_path, client)
    assert result.success is False and result.attempts == 0
    assert client.launched == []
    assert "PreflightError" in result.error
    assert types(events) == ["run_started", "run_failed"]


def test_poll_timeout_stops_job_and_fails(tmp_path):
    client = FakeJobClient([{"state": "RUNNING", "logs": "starting training\nwarming up"}])
    result, _ = do_run(tmp_path, client, max_wait_s=0.05)
    assert result.success is False
    assert client.stopped == ["job-0"]
    assert "poll timeout" in result.error


def test_job_failure_without_playbook_match(tmp_path):
    client = FakeJobClient([{"state": "FAILED", "logs": "kernel exploded mysteriously"}])
    result, _ = do_run(tmp_path, client)
    assert result.success is False
    assert "without a playbook failure match" in result.error


def test_invalid_config_raises_before_any_event(tmp_path):
    events = []
    with pytest.raises(ConfigurationError):
        run(
            {}, dry_run=False, on_event=events.append, runs_dir=str(tmp_path),
            job_client=FakeJobClient([]), poll_interval_s=0.0,
        )
    assert events == []
```

- [ ] **Step 5: replace stale Gate-1 tests (CRITICAL — otherwise a unit test can launch a paid job)**

Two tests assert the old "not implemented (Gate 1)" stub. After Step 3 they would run the
REAL path — and on this machine preflight passes, so `test_gate1…` would hit the live
Nebius API from the unit suite. Replace both:

In `tests/test_runner.py`, replace `test_real_jobs_not_implemented_returns_failed_result` (lines 142–147) with:

```python
def test_real_mode_preflight_failure_returns_run_failed(tmp_path):
    from supertaco.errors import PreflightError
    from supertaco.nebius.jobs import NebiusJobClient

    events, on_event = _collect()

    def _blocked(self):
        raise PreflightError("DNS hijack: 90.207.238.183 (test)")

    orig = NebiusJobClient.preflight
    NebiusJobClient.preflight = _blocked
    try:
        result = run(HEALTHY, dry_run=False, on_event=on_event, llm=FakeLLM(), runs_dir=tmp_path)
    finally:
        NebiusJobClient.preflight = orig
    assert result.success is False
    assert "PreflightError" in result.error
    assert [e.type for e in events] == ["run_started", "run_failed"]
```

(The suite's style is monkeypatch-free at module level here; if `monkeypatch` fixture is
already used elsewhere in the file, prefer it over the orig/try-finally.)

In `tests/test_cli.py`, replace `test_gate1_real_job_prints_failure_and_exits_one` (lines 56–70) with:

```python
def test_real_mode_preflight_failure_prints_failure_and_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # keep runs_dir out of the repo
    fixture = next(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*")
        if "_patched" not in p.name
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: object())

    from supertaco.errors import PreflightError
    from supertaco.nebius.jobs import NebiusJobClient

    def _blocked(self):
        raise PreflightError("preflight blocked (test)")

    monkeypatch.setattr(NebiusJobClient, "preflight", _blocked)
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(fixture), dry_run=False)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "[run_failed]" in out
    assert "PreflightError" in out
    assert "Run failed" in out
```

The dashboard test at `tests/test_dashboard.py:174` only renders a stored error string
(no launch path) — leave it; Task 15 refreshes its wording.

- [ ] **Step 6: gates**

```powershell
python -m pytest tests/test_runner_real.py -q                # expected: all passed
python -m pytest tests/ -m "not integration" -q              # expected: green (sim suite untouched)
python -m ruff check . ; python -m ruff format --check .     # expected: Clean! / unchanged
python -m mypy supertaco/runner.py                           # expected: no new errors
```

- [ ] **Step 7: commit**

```powershell
git add supertaco/runner.py tests/test_runner_real.py tests/test_runner.py tests/test_cli.py docs/PROJECT.md
git commit -m "feat: real-mode runner — monolithic job loop, eval gate, preflight/transport failure handling"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: runner: real branch (_run_real + _poll_until_done, eval gate -> EVAL_REGRESSION, RunResult.eval_results) replacing Gate-1 stub (real-gpu-autoheal plan Task 7).
```

---

### Task 8: ⏸ APPROVAL GATE — live smoke: one real train+eval job (milestone 2)

**Files:**
- Create: `tests/test_jobs_smoke.py`
- Create: `tests/fixtures/` (job_smoke.log tail, responses_smoke.json, job_status_smoke.json)
- Modify: `docs/superpowers/specs/2026-09-28-jobs-contract-probe.md`
- Modify: `docs/PROJECT.md`

**Cost: ~$3–8, ~10–20 min GPU time (L40S). Ask the user for explicit approval before
Step 3; record the approval in `docs/PROJECT.md`.** Dry-run equivalents stay available
(AGENTS rules 2/6): `nebius ai job create --dry-run …` re-run anytime for free.

- [ ] **Step 1: free sanity (no approval needed)**

```powershell
wsl -d kali-linux -- /home/Zephyr/.nebius/bin/nebius ai job list --format json
```

Expected: `{}` (HTTP 200). Also `wsl -d kali-linux -- grep -c nebius /etc/hosts` → ≥13.

- [ ] **Step 2: create `tests/test_jobs_smoke.py`**

```python
"""PAID live smoke: one real train+eval job (milestone 2). ~$3-8.

Opt in after explicit approval:
  $env:SUPERTACO_ALLOW_PAID="1"; python -m pytest tests/test_jobs_smoke.py -m integration -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from supertaco.agent.playbook import detect_failure
from supertaco.eval.harness import DEFAULT_PROMPTS, split_responses
from supertaco.nebius.jobscripts import EVAL_SCRIPT, PREPARE_SCRIPT
from supertaco.nebius.jobs import NebiusJobClient
from supertaco.nebius.recipe import build_job_spec
from supertaco.runner import _poll_until_done

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"
SMOKE_CFG = {
    "learning_rate": 0.0002,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "num_epochs": 1,
}


def test_real_smoke_job() -> None:
    """Launch, poll, and pin contracts for one real job; capture offline fixtures."""
    if os.environ.get("SUPERTACO_ALLOW_PAID") != "1":
        pytest.skip("paid test — set SUPERTACO_ALLOW_PAID=1 after explicit approval")

    client = NebiusJobClient()
    client.preflight()
    prepared = build_job_spec(
        SMOKE_CFG,
        prompts=list(DEFAULT_PROMPTS),
        prepare_script=PREPARE_SCRIPT,
        eval_script=EVAL_SCRIPT,
    )
    launched = client.launch(prepared)
    assert launched.id

    logs, final_state, timed_out = _poll_until_done(
        client, launched.id, 1, None, 20.0, 3600.0
    )
    assert not timed_out, "job did not reach terminal state within 60 min"
    assert final_state == "SUCCEEDED", f"state={final_state}\n{logs[-4000:]}"

    train_logs, responses = split_responses(logs)
    assert responses is not None, "responses marker block missing after success"
    for bucket in ("base", "fine_tuned"):
        assert len(responses[bucket]) == 5
        assert all(v.strip() for v in responses[bucket].values())
    assert detect_failure(train_logs) is None

    # Pin the real state shape: parser must agree with the live poll result.
    args = ["ai", "job", "get", "--id", launched.id, "--format", "json"]
    raw = client._json(client._run(args), args)
    parsed = NebiusJobClient._parse_job_state(raw)
    assert parsed.id == launched.id
    assert parsed.state == final_state

    FIXTURES.mkdir(exist_ok=True)
    # Keep the last 100KB: the responses block is emitted last (eval runs at the
    # end), so the tail always retains it while capping repo size.
    blob = logs.encode("utf-8")
    if len(blob) > 100_000:
        blob = b"...(truncated head)...\n" + blob[-100_000:]
    (FIXTURES / "job_smoke.log").write_bytes(blob)
    (FIXTURES / "responses_smoke.json").write_text(
        json.dumps(responses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Privacy-safe: store only the status field, never the full metadata blob.
    (FIXTURES / "job_status_smoke.json").write_text(
        json.dumps(raw.get("status"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
```

- [ ] **Step 3: ⏸ APPROVAL GATE — ask the user**

Present: *"Live smoke will launch 1 real Nebius L40S job (~10–20 min, ~$3–8 of the $100
credit). Dry-run alternative available. Approve? y/n"*. Do not run Step 4 without a yes;
record the yes + timestamp in `docs/PROJECT.md`.

- [ ] **Step 4: run the smoke (after approval)**

```powershell
$env:SUPERTACO_ALLOW_PAID="1"
python -m pytest tests/test_jobs_smoke.py -m integration -s
```

Expected: 1 passed (allow up to ~25 min — set the tool timeout accordingly).

- [ ] **Step 5: pin observed contracts**

- `assert parsed.state == final_state` failed or state casing differs from
  `TERMINAL_STATES` → fix `NebiusJobClient._parse_job_state`, its unit test
  (Task 4), and this test; record the observed raw status value.
- `--inject-file`/flags rejected → fix `recipe.build_job_spec` + `jobs.launch`
  flags + contract doc + affected tests (probe contract lives in the doc).
- Update `docs/superpowers/specs/2026-09-28-jobs-contract-probe.md` §response shapes
  with the exact observed JSON (redact ids beyond a sample prefix if desired).

- [ ] **Step 6: gates + commit**

```powershell
python -m pytest tests/ -m "not integration" -q   # offline suite still green
python -m ruff check . ; python -m ruff format --check .
git add tests/test_jobs_smoke.py tests/fixtures docs/superpowers/specs/2026-09-28-jobs-contract-probe.md docs/PROJECT.md
git commit -m "test: live smoke — one real train+eval job, pinned state shape, captured fixtures"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: nebius: LIVE SMOKE PASS — 1 real L40S job SUCCEEDED, markers+responses+state shape pinned, fixtures captured; approval granted by user (real-gpu-autoheal plan Task 8).
```

---

### Task 9: ⏸ APPROVAL GATE — failure drill: real auto-heal end-to-end (milestone 3)

**Files:**
- Create: `tests/test_heal_drill.py`
- Modify: `docs/PROJECT.md`

**Cost: up to 4 real jobs (~$6–15 total). Same approval protocol as Task 8.** Proves the
actual promise: a real broken config gets classified from REAL logs, patched, relaunched,
and succeeds — with the real Nemotron judge path untouched (eval wiring is Task 12; this
drill focuses on failure→patch→relaunch).

- [ ] **Step 1: create `tests/test_heal_drill.py`**

```python
"""PAID failure drill: broken config -> real logs -> classify -> patch -> relaunch.

Up to 4 real jobs (~$6-15). Opt in after explicit approval:
  $env:SUPERTACO_ALLOW_PAID="1"; python -m pytest tests/test_heal_drill.py -m integration -s
"""

from __future__ import annotations

import os

import pytest

from supertaco.runner import make_llm, run

pytestmark = pytest.mark.integration

# Deliberately unhealthy on L40S: extreme lr + batch -> NaN and/or OOM in-job.
BROKEN_CFG = {
    "learning_rate": 2.0,
    "batch_size": 128,
    "lora_r": 64,
    "lora_alpha": 128,
    "num_epochs": 3,
}


def test_real_autoheal_drill(tmp_path) -> None:
    if os.environ.get("SUPERTACO_ALLOW_PAID") != "1":
        pytest.skip("paid test — set SUPERTACO_ALLOW_PAID=1 after explicit approval")

    events = []
    result = run(
        BROKEN_CFG,
        dry_run=False,
        max_retries=2,  # initial + 2 relaunches, budget-capped (AGENTS: 3 retries)
        on_event=events.append,
        runs_dir=str(tmp_path),
        llm=make_llm(),
        poll_interval_s=20.0,
        max_wait_s=3600.0,
    )
    kinds = [e.type for e in events]
    failure_keys = [e.data.get("failure_key") for e in events if e.type == "failure_detected"]

    assert "failure_detected" in kinds, f"broken config never failed: {kinds}"
    assert "patch_written" in kinds, "no patch was produced"
    assert "retry_scheduled" in kinds, "no relaunch was scheduled"
    assert result.attempts >= 2, "heal loop never relaunched"
    assert result.success is True, f"auto-heal did not converge: {result.error}"
    assert result.configs_written, "no patched config on disk"
    assert result.final_config != BROKEN_CFG
    print("drill summary:", failure_keys, "->", result.final_config)
```

- [ ] **Step 2: ⏸ APPROVAL GATE — ask the user**

Present: *"Failure drill launches up to 4 real jobs (attempt fails, patch, relaunch;
~10–20 min each, ~$6–15 total). Approve? y/n"*. Record yes + timestamp in PROJECT.md.

- [ ] **Step 3: run (after approval)**

```powershell
$env:SUPERTACO_ALLOW_PAID="1"
python -m pytest tests/test_heal_drill.py -m integration -s
```

Expected: 1 passed — `failure_detected` from REAL job logs (NAN_LOSS/OOM/…), ≥1 patch,
relaunch, converged. **If it does not converge within 2 relaunches:** do NOT keep re-running.
Adjust `BROKEN_CFG` once (e.g. `batch_size: 256` for a certain OOM, or `learning_rate: 5.0`),
get fresh approval for the extra jobs, rerun. If it still fails, record the real behavior in
PROJECT.md and surface it to the user — do not weaken the assertions silently.

- [ ] **Step 4: gates + commit**

```powershell
python -m pytest tests/ -m "not integration" -q
python -m ruff check . ; python -m ruff format --check .
git add tests/test_heal_drill.py docs/PROJECT.md
git commit -m "test: live failure drill — real broken config healed by classify/patch/relaunch loop"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: runner: LIVE FAILURE DRILL PASS — broken config detected from real logs, patched, relaunched, converged; approvals granted (real-gpu-autoheal plan Task 9).
```

---

### Task 10: Offline tests over captured real-job fixtures

**Files:**
- Create: `tests/test_job_fixtures.py`
- Modify (conditional): `supertaco/agent/simlogs.py` (only if loss-point parsing misses real log format)
- Modify: `docs/PROJECT.md`

Tasks 8/9 fixtures make the real-job evidence testable offline forever (no more GPU spend).

- [ ] **Step 1: create `tests/test_job_fixtures.py`**

```python
"""Offline tests over fixtures captured from live Nebius runs (Tasks 8/9)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from supertaco.agent.playbook import detect_failure
from supertaco.agent.simlogs import extract_loss_points
from supertaco.eval.harness import RESPONSES_END, RESPONSES_START, split_responses
from supertaco.nebius.jobs import TERMINAL_STATES, NebiusJobClient

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_files_present():
    for name in ("job_smoke.log", "responses_smoke.json", "job_status_smoke.json"):
        assert (FIXTURES / name).exists(), f"missing fixture {name} — run Task 8 smoke"


def test_real_responses_roundtrip_through_markers():
    responses = json.loads((FIXTURES / "responses_smoke.json").read_text(encoding="utf-8"))
    assert set(responses) == {"base", "fine_tuned"}
    assert all(len(responses[b]) == 5 for b in ("base", "fine_tuned"))
    blob = RESPONSES_START + "\n" + json.dumps(responses) + "\n" + RESPONSES_END
    train_logs, parsed = split_responses("real train line\n" + blob)
    assert parsed == responses
    assert detect_failure(train_logs) is None


def test_real_log_tail_has_no_spurious_failure():
    text = (FIXTURES / "job_smoke.log").read_text(encoding="utf-8", errors="replace")
    train_logs, _ = split_responses(text)
    assert detect_failure(train_logs) is None, "real healthy log misdetected"


def test_real_log_loss_points_parse():
    text = (FIXTURES / "job_smoke.log").read_text(encoding="utf-8", errors="replace")
    points = extract_loss_points(text)
    assert isinstance(points, list)
    if not points:
        pytest.fail(
            "extract_loss_points found nothing in the real job log — extend it to parse "
            "the LLaMA-Factory log format (e.g. {'loss': 0.123, ...}) with a unit test"
        )


def test_status_fixture_parses_to_terminal_state():
    status = json.loads((FIXTURES / "job_status_smoke.json").read_text(encoding="utf-8"))
    parsed = NebiusJobClient._parse_job_state({"status": status, "metadata": {"id": "smoke"}})
    assert parsed.id == "smoke"
    assert parsed.state in TERMINAL_STATES


def test_gen_responses_on_real_log(tmp_path):
    log = FIXTURES / "job_smoke.log"
    if RESPONSES_START.encode() not in log.read_bytes():
        pytest.skip("fixture tail lacks marker block")
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, "scripts/gen_responses.py", str(log), "--out", str(out)],
        capture_output=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    saved = json.loads(out.read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "responses_smoke.json").read_text(encoding="utf-8"))
    assert set(saved) == set(expected) == {"base", "fine_tuned"}
```

- [ ] **Step 2: gates**

```powershell
python -m pytest tests/test_job_fixtures.py -q   # expected: all passed (uses Task 8 fixtures)
python -m pytest tests/ -m "not integration" -q  # expected: green
python -m ruff check . ; python -m ruff format --check .
```

If `test_real_log_loss_points_parse` fails: extend `extract_loss_points` in
`supertaco/agent/simlogs.py` to also parse LLaMA-Factory dict-style lines
(`{'loss': 0.123, ...}`), add a unit test with a captured line, rerun gates.

- [ ] **Step 3: commit**

```powershell
git add tests/test_job_fixtures.py docs/PROJECT.md
git commit -m "test: offline fixture tests over captured real-job logs/status/responses"
```

(plus `supertaco/agent/simlogs.py` if Step 2 needed the parser extension).

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: eval: offline tests over real-job fixtures (roundtrip, no spurious detect, state parse) (real-gpu-autoheal plan Task 10).
```

---

### Task 11: Checkpoint — full gates + user sync before UI/eval phase

No code changes. Confirm the engine works before wiring the UI.

- [ ] **Step 1: full offline gates**

```powershell
python -m pytest tests/ -m "not integration" -q   # expected: all passed
python -m ruff check .                            # expected: Clean!
python -m ruff format --check .                   # expected: unchanged
python -m mypy supertaco/                         # expected: no NEW errors
git status --porcelain                            # expected: clean (after Task 10 commit)
git diff --stat configs/base                      # expected: empty (AGENTS rule 3)
```

- [ ] **Step 2: dry-run still default (AGENTS rule 2 / D8)**

```powershell
python -m supertaco.cli run configs/runs/20260923_191506_NAN_LOSS_broken_model.yaml --dry-run
```

Expected: completes with simulation events, zero Nebius traffic.

- [ ] **Step 3: report to the user**

Summarize: Tasks 1–10 done, live smoke + drill evidence (from PROJECT.md), remaining scope
(Tasks 12–17: judge wiring, D6 flag, CLI/dashboard, demo, docs), current spend vs $100.
Ask: **proceed to the eval/UI phase?** Wait for confirmation.

---

### Task 12: Judge-backed `eval_fn` factory + eval-skip notes

**Files:**
- Modify: `supertaco/eval/harness.py` (add `make_eval_fn`)
- Modify: `supertaco/runner.py` (note when no eval_fn is wired)
- Modify: `tests/test_eval_signal.py`, `tests/test_runner_real.py`
- Modify: `docs/PROJECT.md`

The runner already has the eval seam (Task 7). This task builds the judge-backed factory
(D2: real Nemotron judge, **no hash fallback allowed** in real mode) and makes
"eval not wired" visible instead of silent.

- [ ] **Step 1: `make_eval_fn` in `supertaco/eval/harness.py`**

```python
def make_eval_fn(llm: Any):
    """Judge-backed eval_fn for real mode (D2: real judge only, no hash fallback).

    Returns a callable compatible with runner.run(eval_fn=...): it scores the
    job's base vs fine-tuned responses with run_eval_suite and returns its dict.
    """
    if llm is None:
        raise ValueError("real-mode eval requires an llm judge (D2: no hash fallback)")

    def _eval(responses: dict, prompts: list[str]) -> dict:
        return run_eval_suite(prompts, responses["base"], responses["fine_tuned"], llm=llm)

    return _eval
```

- [ ] **Step 2: runner — visible note when eval_fn is absent**

In `supertaco/runner.py` `_run_real`, healthy path becomes:

```python
            if responses is None:
                eval_note = "responses JSON missing/unparseable — eval skipped (spec 5.5)"
            elif eval_fn is not None:
                eval_results = eval_fn(responses, prompts)
                if eval_results.get("regression_flagged"):
                    failure_key = "EVAL_REGRESSION"  # D1: heal on the shared budget
            else:
                eval_note = "no eval_fn wired — eval skipped"
```

- [ ] **Step 3: tests**

Append to `tests/test_eval_signal.py`:

```python
import pytest

from supertaco.eval.harness import DEFAULT_PROMPTS, make_eval_fn


class ScoringLLM:
    """Judge seam fake: scores by echoing 3/10 for base text, 9/10 for ft text."""

    def __init__(self):
        self.call_log = [{"mode": "real"}]

    def _call(self, key, prompt, max_tokens=None, temperature=None):
        return "3/10" if "BASE ANSWER" in prompt else "9/10"


def _responses(base_text, ft_text):
    return {
        "base": {p: base_text for p in DEFAULT_PROMPTS},
        "fine_tuned": {p: ft_text for p in DEFAULT_PROMPTS},
    }


def test_make_eval_fn_requires_llm():
    with pytest.raises(ValueError, match="D2"):
        make_eval_fn(None)


def test_make_eval_fn_judge_scores_no_regression():
    results = make_eval_fn(ScoringLLM())(_responses("BASE ANSWER", "FT ANSWER"), list(DEFAULT_PROMPTS))
    assert results["regression_flagged"] is False
    assert results["fine_tuned_scores"][0] == 9.0
    assert results["base_scores"][0] == 3.0
    assert "real" in results["modes"][0]


def test_make_eval_fn_detects_regression():
    results = make_eval_fn(ScoringLLM())(_responses("FT ANSWER", "BASE ANSWER"), list(DEFAULT_PROMPTS))
    assert results["regression_flagged"] is True
```

Append to `tests/test_runner_real.py`:

```python
def test_eval_note_when_no_eval_fn(tmp_path):
    client = FakeJobClient(
        [{"state": "SUCCEEDED", "logs": OK_LOGS + "\n" + responses_block(TRICKY)}]
    )
    result, events = do_run(tmp_path, client)  # eval_fn not provided
    assert result.success is True and result.eval_results is None
    ev = next(e for e in events if e.type == "run_succeeded")
    assert "no eval_fn" in ev.data["eval_note"]
```

- [ ] **Step 4: gates + commit**

```powershell
python -m pytest tests/ -m "not integration" -q   # expected: green
python -m ruff check . ; python -m ruff format --check .
python -m mypy supertaco/eval/harness.py supertaco/runner.py
git add supertaco/eval/harness.py supertaco/runner.py tests/test_eval_signal.py tests/test_runner_real.py docs/PROJECT.md
git commit -m "feat: make_eval_fn judge factory (D2) + visible eval-skip notes"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: eval: make_eval_fn (judge-backed, D2 no-hash) + no-eval_fn visibility note (real-gpu-autoheal plan Task 12).
```

---

### Task 13: D6 — mode-aware regression/health flag

**Files:**
- Modify: `supertaco/eval/harness.py` (add `display_regression_flag`)
- Modify: `tests/test_eval_signal.py`
- Modify: `docs/PROJECT.md`

D6: in simulation/dry-run the displayed regression flag is **config health ground
truth** (`classify_config_failure`), not simulated score floors; in real mode it is the
judge's ratio verdict.

- [ ] **Step 1: helper in `supertaco/eval/harness.py`**

```python
def display_regression_flag(
    mode: str, eval_results: dict | None, final_config: dict
) -> tuple[bool, str]:
    """Displayed regression flag (D6): simulation = config health, real = judge verdict.

    Returns (flagged, human_note). Simulation never trusts simulated scores;
    real mode never falls back to config heuristics (D9) — eval skipped means
    no regression evidence, not "healthy".
    """
    if mode == "real":
        if not eval_results:
            return False, "eval skipped — no regression evidence (spec 5.5)"
        if eval_results.get("regression_flagged"):
            return True, "judge: fine-tuned average >10% below baseline"
        return False, "judge: fine-tuned within baseline"
    from supertaco.agent.simlogs import classify_config_failure

    if classify_config_failure(final_config) is not None:
        return True, "config unhealthy (ground truth, D6)"
    return False, "config healthy (ground truth, D6)"
```

- [ ] **Step 2: tests — append to `tests/test_eval_signal.py`**

```python
from supertaco.eval.harness import display_regression_flag


def test_d6_simulation_uses_config_health_not_scores():
    healthy = {"learning_rate": 0.0001, "batch_size": 8}
    floored = {"regression_flagged": True, "base_scores": [9.0] * 5, "fine_tuned_scores": [1.0] * 5}
    flagged, note = display_regression_flag("simulation", floored, healthy)
    assert flagged is False          # pins D6: floored scores must not flag healthy configs
    assert "healthy" in note
    broken = {"learning_rate": 0.05, "batch_size": 8, "num_epochs": 3}
    flagged, note = display_regression_flag("simulation", None, broken)
    assert flagged is True
    assert "unhealthy" in note


def test_d6_real_uses_judge_verdict_only():
    flagged, note = display_regression_flag("real", {"regression_flagged": True}, {})
    assert flagged is True and "judge" in note
    flagged, _ = display_regression_flag("real", {"regression_flagged": False}, {"batch_size": 64})
    assert flagged is False          # real mode ignores config heuristics (D9)
    flagged, note = display_regression_flag("real", None, {})
    assert flagged is False and "eval skipped" in note


def test_d6_mode_strings_exact():
    # callers pass exactly "real" or "simulation"; anything else falls to the
    # simulation branch (documented contract — dashboard/CLI own the mapping)
    flagged, _ = display_regression_flag("anything", {"regression_flagged": True}, {"batch_size": 64})
    assert flagged is True  # broken config, simulation branch
```

- [ ] **Step 3: gates + commit**

```powershell
python -m pytest tests/ -m "not integration" -q
python -m ruff check . ; python -m ruff format --check .
git add supertaco/eval/harness.py tests/test_eval_signal.py docs/PROJECT.md
git commit -m "feat: display_regression_flag — D6 mode-aware regression/health semantics"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: eval: display_regression_flag (D6: sim=config health ground truth, real=judge) (real-gpu-autoheal plan Task 13).
```

---

### Task 14: CLI `--real` — confirmation gate, judge wiring, run history

**Files:**
- Modify: `supertaco/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/PROJECT.md`

AGENTS rule 2/6: simulation stays the default; `--real` is explicit, interactively
confirmed (or `--yes` for scripts), judge-wired, and every run is recorded with its
approval mode.

- [ ] **Step 1: `supertaco/cli.py`**

Imports: add `from datetime import UTC, datetime` and `from pathlib import Path`.

Parser (replace the single `--dry-run` arg):

```python
    run_parser = subparsers.add_parser("run", help="Run a fine-tuning job")
    run_parser.add_argument("config", help="Path to YAML config file")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Simulate without launching real jobs (default behavior)",
    )
    mode.add_argument(
        "--real", action="store_true",
        help="Launch REAL Nebius GPU jobs (spends credit, asks for confirmation)",
    )
    run_parser.add_argument(
        "--yes", action="store_true",
        help="Skip the real-run confirmation prompt (for scripts/CI)",
    )

    args = parser.parse_args()
    if args.command == "run":
        run_config(args.config, dry_run=not args.real, assume_yes=args.yes)
    else:
        parser.print_help()
        sys.exit(0)
```

Replace `run_config` and add helpers:

```python
def run_config(
    config_path: str,
    dry_run: bool = True,
    assume_yes: bool = False,
    history_path: str = "logs/run_history.jsonl",
) -> None:
    """Run a fine-tuning job from a config file via the shared runner.

    Real mode (--real / dry_run=False) requires confirmation (or assume_yes),
    wires the judge-backed eval_fn, and records the approval in the run history.
    """
    try:
        config = _load_config(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error loading {config_path}: {exc}") from exc
    if not dry_run and not assume_yes and not _confirm_spend():
        raise SystemExit("Aborted — no Nebius credit spent.")
    print(f"SuperTaco: running {config_path} ({'simulation' if dry_run else 'REAL GPU'})")
    llm = _make_llm()

    def printer(event) -> None:
        print(f"[{event.type}] {json.dumps(event.data, default=str)}")

    eval_fn = None
    if not dry_run:
        from supertaco.eval.harness import make_eval_fn

        eval_fn = make_eval_fn(llm)  # D2: judge-backed, refuses llm=None

    try:
        result = runner_run(
            config, max_retries=3, dry_run=dry_run, on_event=printer, llm=llm, eval_fn=eval_fn
        )
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    approval = "not-needed" if dry_run else ("cli-yes-flag" if assume_yes else "cli-interactive")
    _record_run(config_path, dry_run, approval, result, history_path)
    if result.success:
        print(f"Success after {result.attempts} attempt(s). Configs: {result.configs_written}")
        if result.eval_results:
            print(f"Eval: {json.dumps(result.eval_results, default=str)}")
    else:
        print(f"Run failed: {result.error}")
        sys.exit(1)


def _confirm_spend() -> bool:
    """Interactive approval for cost-bearing runs (AGENTS rule 6)."""
    try:
        answer = input("REAL Nebius GPU jobs will spend credit. Type 'yes' to continue: ")
    except EOFError:
        return False
    return answer.strip().lower() == "yes"


def _record_run(
    config_path: str, dry_run: bool, approval: str, result, history_path: str
) -> None:
    """Append one JSONL record per run, including the approval mode."""
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "mode": "simulation" if dry_run else "real",
        "approval": approval,
        "success": result.success,
        "attempts": result.attempts,
        "failure_key": result.failure_key,
        "error": result.error,
        "eval_results": result.eval_results,
        "configs_written": result.configs_written,
    }
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
```

- [ ] **Step 2: tests — `tests/test_cli.py`**

Update the Task-7 preflight test call: `cli.run_config(str(fixture), dry_run=False, assume_yes=True)`
(and its name/assertions stay). Append:

```python
import json
import sys


def test_real_mode_blocked_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture = next(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*")
        if "_patched" not in p.name
    )
    monkeypatch.setattr(cli, "_confirm_spend", lambda: False)
    monkeypatch.setattr(cli, "_make_llm", lambda: object())
    with pytest.raises(SystemExit) as exc:
        cli.run_config(str(fixture), dry_run=False, assume_yes=False)
    assert "Aborted" in str(exc.value)
    assert not (tmp_path / "logs" / "run_history.jsonl").exists()  # nothing ran


def test_real_and_dry_run_flags_conflict(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["supertaco", "run", "x.yaml", "--real", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2  # argparse usage error


def test_history_record_written_for_simulation(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fixture = next(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*")
        if "_patched" not in p.name
    )

    class FakeLLM:
        call_log = []

        def classify_failure(self, logs):
            return "NAN_LOSS"

        def propose_patch(self, key, config, logs):
            return {"fix": "x", "reason": "y"}

    monkeypatch.setattr(cli, "_make_llm", lambda: FakeLLM())
    cli.run_config(str(fixture), dry_run=True)
    lines = (tmp_path / "logs" / "run_history.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    assert rec["mode"] == "simulation"
    assert rec["approval"] == "not-needed"
    assert rec["success"] is True
    assert rec["attempts"] >= 1


def test_real_mode_wires_eval_fn_and_records_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fixture = next(
        p
        for p in (Path(__file__).resolve().parents[1] / "configs" / "runs").glob("*NAN_LOSS*")
        if "_patched" not in p.name
    )
    seen = {}

    def fake_run(config, **kw):
        seen.update(kw)
        from supertaco.runner import RunResult

        return RunResult(True, 1, dict(config), [], None, None, [], eval_results={"regression_flagged": False})

    monkeypatch.setattr(cli, "runner_run", fake_run)
    monkeypatch.setattr(cli, "_make_llm", lambda: object())
    cli.run_config(str(fixture), dry_run=False, assume_yes=True)
    assert seen["dry_run"] is False
    assert seen["eval_fn"] is not None  # judge factory wired (D2)
    lines = (tmp_path / "logs" / "run_history.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    assert rec["mode"] == "real"
    assert rec["approval"] == "cli-yes-flag"
    assert rec["eval_results"] == {"regression_flagged": False}
    out = capsys.readouterr().out
    assert "REAL GPU" in out and "Eval:" in out
```

- [ ] **Step 3: gates**

```powershell
python -m pytest tests/test_cli.py -q               # expected: all passed
python -m pytest tests/ -m "not integration" -q     # expected: green
python -m ruff check . ; python -m ruff format --check .
python -m mypy supertaco/cli.py                     # expected: no new errors
```

- [ ] **Step 4: commit**

```powershell
git add supertaco/cli.py tests/test_cli.py docs/PROJECT.md
git commit -m "feat: CLI --real with confirmation gate, judge eval wiring, JSONL run history"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: cli: --real + --yes confirmation gate, make_eval_fn wiring, logs/run_history.jsonl approval records (real-gpu-autoheal plan Task 14).
```

---

### Task 15: Dashboard — mode toggle, real wiring, D6 flag, payload rendering

**Files:**
- Modify: `supertaco/ui/dashboard.py`
- Modify: `tests/test_dashboard.py`
- Modify: `docs/PROJECT.md`

- [ ] **Step 1: sidebar — mode radio + cost acknowledgement**

In `render_sidebar()`, insert before the `### Max Retries` section:

```python
    st.sidebar.markdown("### Mode")
    st.sidebar.radio(
        "Launch mode",
        ["Simulation", "Real GPU"],
        index=0,
        key="mode",
        help="Simulation is free and default. Real GPU spends Nebius credit (~$3-15/run).",
    )
    if st.session_state.get("mode") == "Real GPU":
        st.sidebar.warning("⚠ Real GPU launches paid Nebius jobs ($100 credit budget).")
        st.sidebar.checkbox(
            "I understand this launches real Nebius jobs",
            key="real_ack_box",
        )
```

- [ ] **Step 2: launch block — replace the dry-run checkbox**

Replace the `c1` checkbox (current lines 411–414) with:

```python
        with c1:
            mode = st.session_state.get("mode", "Simulation")
            dry_run_on = mode == "Simulation"
            if dry_run_on:
                st.success("Mode: Simulation (free — no Nebius traffic)")
            else:
                st.warning("Mode: REAL GPU — spends Nebius credit")
                if not st.session_state.get("real_ack_box", False):
                    st.caption("Tick the sidebar checkbox to enable launch.")
```

And the launch button becomes:

```python
        run_clicked = st.button(
            "▶ Launch Job",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.get("mode") == "Real GPU"
                and not st.session_state.get("real_ack_box", False)
            ),
        )
```

`dry_run=dry_run_on` in the `execute_launch(...)` call stays as-is.

- [ ] **Step 3: `execute_launch` — judge wiring for real mode**

```python
def execute_launch(
    config: dict,
    max_retries: int,
    timeline_slot,
    logs_slot,
    chart_slot,
    dry_run: bool = True,
):
    """Run the pipeline, streaming events into the status placeholders."""
    events: list[Event] = []

    def on_event(ev: Event) -> None:
        events.append(ev)
        st.session_state["events"] = events  # survive a mid-run crash (spec 6)
        timeline_slot.markdown(_render_events(events))
        if ev.type == "logs_produced":
            logs_slot.code(ev.data["text"], language="log")
            chart_slot.plotly_chart(_loss_figure(events), use_container_width=True)

    llm = make_llm()
    eval_fn = None
    if not dry_run:
        from supertaco.eval.harness import make_eval_fn

        eval_fn = make_eval_fn(llm)  # D2: judge-backed, refuses llm=None
    result = runner_run(
        config,
        max_retries=max_retries,
        on_event=on_event,
        llm=llm,
        dry_run=dry_run,
        eval_fn=eval_fn,
    )
    return events, result
```

- [ ] **Step 4: `_render_events` — support the real job_launched payload**

Replace the `job_launched` branch (current lines 289–293) with:

```python
        if ev.type == "job_launched":
            p = d["payload"]
            if "config" in p:  # simulation payload
                lines.append(
                    f"**Attempt {d['attempt']}** · 🚀 dry-run launch "
                    f"({len(p['config'])} config keys)"
                )
            else:  # real payload (Task 7)
                lines.append(
                    f"**Attempt {d['attempt']}** · 🚀 real launch `{p.get('name', '?')}` "
                    f"· {p.get('platform', '')} · {p.get('image', '')}"
                )
```

- [ ] **Step 5: `render_eval_panel` — real-run results + D6 flag**

Replace the results lookup and the regression line (current lines 253–269) with:

```python
    mode = "real" if st.session_state.get("mode") == "Real GPU" else "simulation"
    results = st.session_state.get("eval_results")
    if mode == "real" and getattr(result, "eval_results", None):
        results = result.eval_results  # judge scores from the real run win (D2)
    if not results:
        note = next(
            (
                e.data.get("eval_note")
                for e in reversed(st.session_state.get("events", []))
                if e.type == "run_succeeded" and e.data.get("eval_note")
            ),
            None,
        )
        if note:
            st.info(f"📊 {note}")
        return

    base_avg = sum(results["base_scores"]) / len(results["base_scores"])
    ft_avg = sum(results["fine_tuned_scores"]) / len(results["fine_tuned_scores"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Before (base)", f"{base_avg:.2f}")
    col2.metric("After (fine-tuned)", f"{ft_avg:.2f}")
    col3.metric("Improvement", f"{ft_avg - base_avg:+.2f}")
    from supertaco.eval.harness import display_regression_flag

    flagged, flag_note = display_regression_flag(mode, results, result.final_config)
    if flagged:
        st.error(f"🚨 EVAL REGRESSION — {flag_note}")
    else:
        st.caption(f"✅ Regression check — {flag_note}")
    if any("hash" in m for m in results["modes"]):
        st.warning(
            "⚠ Judge fell back to hash mode (LLM unavailable) — "
            "scores are deterministic placeholders."
        )
```

(The dataframe block below stays.)

- [ ] **Step 6: tests — `tests/test_dashboard.py`**

Update `test_render_events_shows_run_failed_details` (lines 171–185): change the second
event's error to `"PreflightError: DNS hijack: 90.207.238.183 (test)"` and the assertion
`assert "Gate 1" in md` → `assert "PreflightError" in md`.

Append:

```python
def test_mode_radio_defaults_to_simulation(app):
    radio = next(r for r in app.radio if r.label == "Launch mode")
    assert radio.options == ["Simulation", "Real GPU"]
    assert radio.value == "Simulation"
    launch = next(b for b in app.button if "Launch Job" in b.label)
    assert launch.disabled is False


def test_real_mode_requires_ack_and_shows_cost(app):
    radio = next(r for r in app.radio if r.label == "Launch mode")
    radio.set_value("Real GPU").run()
    assert not app.exception, [e.value for e in app.exception]
    warns = [w.value for w in app.sidebar.warning]
    assert any("credit" in w.lower() for w in warns)
    launch = next(b for b in app.button if "Launch Job" in b.label)
    assert launch.disabled is True
    ack = next(c for c in app.checkbox if "I understand" in c.label)
    ack.set_value(True).run()
    launch = next(b for b in app.button if "Launch Job" in b.label)
    assert launch.disabled is False


def test_eval_panel_d6_simulation_flag_is_config_health(app):
    app = _launch_nan(app)  # ends with a healed (healthy) final config
    eval_button = next(b for b in app.button if "Eval" in b.label)
    eval_button.click().run()
    assert not app.exception, [e.value for e in app.exception]
    err_texts = [e.value for e in app.error]
    assert not any("EVAL REGRESSION" in t for t in err_texts)  # healthy config, D6


def test_render_events_real_launch_payload():
    from supertaco.runner import Event
    from supertaco.ui.dashboard import _render_events

    md = _render_events(
        [
            Event(
                type="job_launched",
                data={
                    "attempt": 1,
                    "payload": {
                        "name": "supertaco-1",
                        "image": "hiyouga/llamafactory:latest",
                        "platform": "gpu-l40s-d",
                        "parent_id": "project-x",
                    },
                },
            )
        ]
    )
    assert "supertaco-1" in md and "gpu-l40s-d" in md
    md2 = _render_events(
        [Event(type="job_launched", data={"attempt": 1, "payload": {"config": {"a": 1, "b": 2}}})]
    )
    assert "2 config keys" in md2


def test_execute_launch_wires_eval_fn_only_in_real_mode(monkeypatch):
    import supertaco.ui.dashboard as dash
    from supertaco.runner import RunResult

    captured = []

    def fake_run(config, **kw):
        captured.append(kw)
        return RunResult(True, 1, dict(config), [], None, None, [])

    monkeypatch.setattr(dash, "make_llm", lambda: object())
    monkeypatch.setattr(dash, "runner_run", fake_run)
    dash.execute_launch({"learning_rate": 0.0001}, 3, None, None, None, dry_run=True)
    dash.execute_launch({"learning_rate": 0.0001}, 3, None, None, None, dry_run=False)
    assert captured[0]["eval_fn"] is None       # simulation: no judge
    assert captured[1]["eval_fn"] is not None   # real: judge factory wired (D2)
    assert captured[1]["dry_run"] is False
```

Note: `execute_launch` real branch calls `make_eval_fn(object())` — non-None llm, so the
factory accepts it; no network happens because `fake_run` never invokes the judge.

- [ ] **Step 7: gates**

```powershell
python -m pytest tests/test_dashboard.py -q      # expected: all passed
python -m pytest tests/ -m "not integration" -q  # expected: green
python -m ruff check . ; python -m ruff format --check .
python -m mypy supertaco/ui/dashboard.py         # expected: no new errors
```

- [ ] **Step 8: commit**

```powershell
git add supertaco/ui/dashboard.py tests/test_dashboard.py docs/PROJECT.md
git commit -m "feat: dashboard mode toggle (Simulation/Real GPU) + ack gate, judge wiring, D6 flag, real payload rendering"
```

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: dashboard: mode radio + real-ack gate, execute_launch judge wiring, D6 eval flag, real job_launched rendering (real-gpu-autoheal plan Task 15).
```

---

### Task 16: ⏸ APPROVAL GATE — full-loop demo (milestone 4)

**Files:**
- Modify: `docs/PROJECT.md` (evidence entry)

The capstone: broken config → real Nebius job(s) → real logs classified → patched →
relaunched → judge-evaluated → success, all through the shipped CLI.

- [ ] **Step 1: ⏸ APPROVAL GATE — ask the user**

Present: *"Demo: `--real` run on the NAN_LOSS fixture — up to 4 real jobs (~$10–20,
~40–80 min). Confirms interactively via the `yes` prompt. Cumulative spend after this
stays well under the $100 budget. Approve? y/n"*. Record yes in PROJECT.md.

- [ ] **Step 2: run the demo (after approval), exercising the interactive gate**

```powershell
python -m supertaco.cli run configs/runs/20260923_191506_NAN_LOSS_broken_model.yaml --real
```

Expected: confirmation prompt → type `yes` → `[run_started]` … `[job_launched]` with a
real payload → `[logs_produced]` → either `[failure_detected]`→patch→relaunch or a
direct success → final `Success after N attempt(s)` + `Eval: {...}` line, exit code 0.

- [ ] **Step 3: verify the run record**

```powershell
Get-Content logs/run_history.jsonl -Tail 1 | python -m json.tool
```

Expected fields: `"mode": "real"`, `"approval": "cli-interactive"`, `"success": true`,
`"eval_results"` a dict with `regression_flagged` + score lists (real judge, modes show
`real`/`fallback` — not `hash` unless Token Factory was down), `"attempts" >= 1`.

- [ ] **Step 4: optional dashboard showcase (manual)**

`streamlit run supertaco/ui/dashboard.py` → sidebar **Real GPU** + ack → load the
NAN_LOSS fixture → Launch → watch real timeline/logs/loss chart → Run Eval Suite shows
the D6 flag. Screenshot for the hackathon submission if desired.

- [ ] **Step 5: record evidence**

Append to `docs/PROJECT.md`:

```markdown
- 2026-09-28: demo: FULL-LOOP DEMO PASS — CLI --real on NAN_LOSS fixture healed on real GPU with judge eval; run_history record verified (approval: cli-interactive); N attempt(s), M job(s); approval granted (real-gpu-autoheal plan Task 16).
```

(Fill N/M from the actual run. If the fixture unexpectedly succeeds first-try, that is a
valid demo outcome — record it as-is. If it fails outright, stop and surface to the user;
do not burn extra jobs without fresh approval.)

---

### Task 17: Docs, milestone sweep, final gates

**Files:**
- Modify: `docs/STACK.md`, `README.md` (if present), `docs/HANDOFF.md`, `docs/CONVENTIONS.md`
- Modify: `docs/superpowers/specs/2026-09-28-real-gpu-autoheal-design.md` (§8 milestone rows)
- Modify: `docs/PROJECT.md`
- No behavior changes.

- [ ] **Step 1: STACK.md**

Add/update a "Real GPU runs" section: Qwen2.5-1.5B-Instruct + `yahma/alpaca-cleaned`
500-row slice + `max_steps=60` + L40S (`gpu-l40s-d`, `1gpu-16vcpu-200gb`, `--timeout 1h`),
transport = WSL `nebius` CLI subprocess (`NEBIUS_CLI`), `--real` usage + `--yes`, cost
guardrails (approval gates, `logs/run_history.jsonl`, $100 budget), mode toggle
(Simulation default), `SUPERTACO_ALLOW_PAID=1` for integration tests.

- [ ] **Step 2: HANDOFF.md**

Replace the "Gate 1 … intentionally unimplemented" wording (lines ~116–123 invariants,
~170–172 Gate 1) with: real loop implemented per `2026-09-28-real-gpu-autoheal-design.md`;
invariants now read: (1) simulation stays default, (2) every real run is confirmed +
recorded, (3) no silent fallbacks (D9) — eval/transport failures are visible notes or
`run_failed`. Keep any still-true invariants unchanged.

- [ ] **Step 3: spec §8 milestone rows**

Mark milestones 1–5 ✅ with one-line evidence each (contract probe doc, smoke fixtures,
drill result, demo record path, dashboard toggle) — milestone 0 already updated in Task 1.

- [ ] **Step 4: README/CONVENTIONS (if they document run modes)**

Add `--real` examples, the approval flow, and the JSONL history location; remove any
"dry-run only / Gate 1" claims.

- [ ] **Step 5: final gates (all must pass before claiming done)**

```powershell
python -m pytest tests/ -m "not integration" -q   # expected: all passed
python -m ruff check .                            # expected: Clean!
python -m ruff format --check .                   # expected: unchanged
python -m mypy supertaco/                         # expected: no NEW errors (baseline debt untouched)
git status --porcelain                            # expected: clean after final commit
git diff --stat configs/base                      # expected: empty (AGENTS rule 3)
git log --oneline -15                             # review the task commits
```

Also confirm secrets never staged: `git diff --cached --name-only` never lists `.env`,
`logs/`, or `*_patched.yaml`.

- [ ] **Step 6: final PROJECT.md entry + commit**

```markdown
- 2026-09-28: docs: real-GPU auto-heal shipped — STACK/HANDOFF/spec §8 updated, final gates green (real-gpu-autoheal plan Task 17).
```

```powershell
git add docs/ README.md docs/PROJECT.md
git commit -m "docs: real GPU auto-heal complete — stack/handoff/milestone updates"
```

Report to the user: tasks done, tests count, spend vs $100, and ask whether to push
(do NOT push without explicit instruction).

---

## Self-review (writing-plans checklist)

- **Scope:** one plan, milestones 0–5, no work outside the spec's decisions D1–D9; sim path
  untouched (D8) and its tests remain authoritative.
- **No placeholders:** every step has copy-pasteable code, exact commands, and expected
  outputs; the only intentional conditionals are approval gates and contract-pinning
  fallbacks (Task 8 Step 5, Task 9 Step 3), each with explicit branch instructions.
- **Cost safety:** paid tasks (8, 9, 16) are ⏸ gates with `SUPERTACO_ALLOW_PAID` opt-in;
  the dangerous stale Gate-1 tests are neutralized in Task 7 Step 5 *before* the real path
  can execute; simulation remains the default everywhere (rule 2).
- **Verification:** per-task gates + fixture-based regression tests (Task 10) + final
  gate battery (Task 17 Step 5); success claims require command output evidence.
- **Interfaces pinned:** recipe/client/runner/eval seams were written against the actual
  repo code (`runner.run` signature, `RunResult` positional construction, `detect_failure`
  patterns incl. the prompt-leak hazard, `score_response`'s `llm._call` seam,
  `classify_config_failure` rules, AppTest conventions).
