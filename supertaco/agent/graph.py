from __future__ import annotations

from typing import Dict, Any, Optional, Literal
from supertaco.agent.playbook import detect_failure, apply_default_fix
from supertaco.agent.llm import NemotronClient
from supertaco.errors import MaxRetriesExceeded


# Type definitions for the state graph
State = Dict[str, Any]


class AgentState:
    """State for the LangGraph agent supervisor loop."""

    def __init__(
        self,
        config: dict,
        logs: str = "",
        failure_key: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        job_id: Optional[str] = None,
        error_history: Optional[list] = None,
    ):
        self.config = config
        self.logs = logs
        self.failure_key = failure_key
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.job_id = job_id
        self.error_history = error_history

    def __repr__(self):
        return f"AgentState(job={self.job_id}, retry={self.retry_count}/{self.max_retries}, failure={self.failure_key})"


def classify_failure_node(state: State) -> State:
    """LangGraph node: classify the failure mode from logs."""
    logs = state.get("logs", "")
    failure_key = detect_failure(logs)
    return {
        **state,
        "failure_key": failure_key,
        "logs": logs,
    }


def propose_patch_node(state: State) -> State:
    """LangGraph node: propose a config patch using the LLM."""
    failure_key = state.get("failure_key")
    config = state.get("config", {})
    logs = state.get("logs", "")

    if not failure_key:
        return {**state, "patch_proposed": False, "error": "No failure detected"}

    # Use LLM to propose patch
    from supertaco.agent.llm import NemotronClient

    client = NemotronClient(
        base_url=config.get("token_factory_base_url", ""),
        api_key=config.get("nebius_api_key", ""),
    )

    patch_response = client.propose_patch(failure_key, config, logs)
    llm_log_token_estimate = len(str(patch_response))

    # Log the LLM call
    client.log_call("patch", llm_log_token_estimate, 1.0, f"proposed patch for {failure_key}")

    # Apply the default fix to create new config
    new_config = apply_default_fix(failure_key, config)

    return {
        **state,
        "patch_proposed": True,
        "patch_response": patch_response,
        "config": new_config,
        "error": None,
    }


def launch_job_node(state: State) -> State:
    """LangGraph node: launch the training job."""
    config = state.get("config", {})
    from supertaco.nebius.jobs import NebiusJobClient

    # In real implementation, would use the actual client
    # For now, simulate job launch
    job_id = f"sim_job_{state.get('retry_count', 0)}"
    logs = state.get("logs", "")

    # Simulate getting logs from the job
    new_logs = logs or f"Job {job_id} started training..."

    return {
        **state,
        "job_id": job_id,
        "logs": new_logs,
        "status": "launched",
        "error": None,
    }


def monitor_node(state: State) -> State:
    """LangGraph node: monitor job status and check for failures."""
    logs = state.get("logs", "")
    failure_key = detect_failure(logs)

    return {
        **state,
        "failure_key": failure_key,
        "logs": logs,
        "error": None if failure_key is None else f"Detected: {failure_key}",
    }


def should_continue(retry_count: int, max_retries: int, failure_key: str | None = None) -> Literal["continue", "stop"]:
    """Determine if the agent loop should continue."""
    if retry_count >= max_retries:
        return "stop"
    if failure_key is None:
        return "stop"
    return "continue"


def run_agent_graph(
    initial_config: dict,
    max_retries: int = 3,
    dry_run: bool = True,
) -> dict:
    """Run the LangGraph agent supervisor loop.

    Executes the monitor → classify → [known: apply fix | unknown: Tavily lookup]
    → patch config → relaunch loop, enforcing the 3-retry cap.

    Returns final state dict.
    """
    state: State = {
        "config": initial_config,
        "logs": "",
        "failure_key": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "job_id": None,
    }

    step = 0
    while step < 50:  # safety limit
        step += 1

        # Monitor: check for failures
        state = monitor_node(state)

        # Check if we should stop
        decision = should_continue(
            retry_count=state["retry_count"],
            max_retries=state["max_retries"],
            failure_key=state.get("failure_key"),
        )
        if decision == "stop":
            break

        # Classify
        state = classify_failure_node(state)

        if state.get("failure_key") is None:
            # No failure, job completed
            break

        # For known failure modes, apply default fix
        # For unknown, do Tavily lookup
        known_modes = [
            "NAN_LOSS", "OOM", "LOSS_DIVERGENCE", "LOSS_PLATEAU",
            "EVAL_REGRESSION", "TOKENIZER_MISMATCH", "DATALOADER_STALL", "GRADIENT_EXPLOSION"
        ]

        if state.get("failure_key") in known_modes:
            # Apply default fix
            state = propose_patch_node(state)
            state = launch_job_node(state)
            state = monitor_node(state)
            state = {
                **state,
                "retry_count": state["retry_count"] + 1,
            }
        else:
            # Unknown failure - Tavily lookup (would be separate node)
            # For now, just break
            break

    return {
        "config": state["config"],
        "job_id": state["job_id"],
        "final_logs": state["logs"],
        "final_failure_key": state.get("failure_key"),
        "retry_count": state["retry_count"],
        "success": state.get("failure_key") is None,
    }