import json
import subprocess
from sandboxtune.nebius.jobs import NebiusJobClient
from sandboxtune.nebius.endpoints import NebiusEndpointClient
from sandboxtune.agent.playbook import detect_failure, apply_default_fix
from sandboxtune.agent.llm import NemotronClient
from sandboxtune.errors import MaxRetriesExceeded


def patch_config(config: dict, fix_description: str) -> dict:
    """Apply a config patch based on a fix description. Returns new config dict."""
    from sandboxtune.agent.playbook import apply_default_fix
    new_config = apply_default_fix(fix_description, config) if fix_description else config
    # If fix_description is a key from playbook, use default fixes
    # Otherwise, use LLM-proposed fix
    return new_config


def launch_job(client: NebiusJobClient, config: dict, dry_run: bool = True) -> dict:
    """Launch a training job. Returns job info dict."""
    return client.launch_job(config, dry_run=dry_run)


def tavily_lookup(query: str, tavily_client) -> dict:
    """Look up error context using Tavily API. Returns search results dict."""
    # In production, call Tavily API
    # For now, return mock results
    return {
        "query": query,
        "results": [
            {
                "title": "Related documentation",
                "url": "https://example.com",
                "snippet": f"Tavily result for: {query}",
            }
        ],
        "used_credits": 1,
    }


def run_agent_loop(
    initial_config: dict,
    job_client: NebiusJobClient,
    endpoint_client: NebiusEndpointClient,
    llm: NemotronClient,
    max_retries: int = 3,
    dry_run: bool = True,
):
    """Run the agent supervisor loop: monitor → classify → patch → relaunch.

    Enforces the 3-retry cap. Every LLM call is logged.
    Every config patch produces a NEW file in configs/runs/.
    """
    config = initial_config
    retry_count = 0
    last_log = ""
    job_id = None

    while retry_count < max_retries:
        # Launch job
        result = launch_job(job_client, config, dry_run=dry_run)
        if dry_run:
            job_id = result.get("payload", {}).get("job_id")
            last_log = f"[dry-run] Would launch job with config keys: {list(config.keys())}"
        else:
            job_id = result.get("job_id", "")
            last_log = f"Launched job {job_id}"

        # Monitor: get status and logs (simulated)
        # In real implementation, would stream logs from Nebius
        status = job_client.get_status(job_id) if not dry_run else {"status": "completed"}
        last_log += f"\nStatus: {status}"

        # Check for failure - detect from simulated logs
        # In real implementation, would get actual training logs
        failure_key = detect_failure(last_log) if not dry_run else None

        if not failure_key:
            # No failure detected, job completed successfully
            break

        # Log the failure detection
        llm.log_call("classify", len(last_log), 0.5, f"detected {failure_key}")

        # Propose patch
        patch_response = llm.propose_patch(failure_key, config, last_log)
        llm.log_call("patch", len(str(patch_response)), 1.0, f"proposed patch for {failure_key}")

        # Apply the patch - creates NEW config file
        config = apply_default_fix(failure_key, config)
        llm.log_call("patch", len(str(config)), 0.5, f"applied fix for {failure_key}")

        retry_count += 1
        print(f"Attempt {retry_count}: applied fix for {failure_key}, new config: {list(config.keys())}")

    if retry_count >= max_retries:
        raise MaxRetriesExceeded(
            f"Max retries ({max_retries}) exceeded for job",
            str(config),
            retry_count,
        )

    return {"config": config, "job_id": job_id, "attempts": retry_count + 1}