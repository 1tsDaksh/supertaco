"""scripts/sweep.py: RESULTS.md rendering + resume/plan gates (WS4 runner)."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP_PATH = REPO / "scripts" / "sweep.py"


def _load_sweep():
    spec = importlib.util.spec_from_file_location("sweep", SWEEP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _artifact(run_id, goal, success=True, attempts=2, tokens=7):
    return {
        "schema": 1,
        "run_id": run_id,
        "goal": goal,
        "config_path": "configs/runs/p.yaml",
        "started": "2026-10-20T12:00:00+00:00",
        "finished": "2026-10-20T12:10:00+00:00",
        "wall_clock_s": 600.0,
        "success": success,
        "attempts": attempts,
        "failure_keys": ["OOM"] if attempts > 1 else [],
        "healed": attempts > 1,
        "final_config": None,
        "judge_backend": "nebius_job",
        "eval": {
            "prompts": ["a"],
            "base_scores": [5.0],
            "fine_tuned_scores": [8.0],
            "base_avg": 5.0,
            "ft_avg": 8.0,
            "delta": 3.0,
            "modes": ["base:real / ft:real"],
            "regression_flagged": False,
        },
        "events": [
            {
                "type": "job_launched",
                "data": {
                    "attempt": 1,
                    "payload": {"config": {}, "model": "Qwen/Qwen2.5-0.5B-Instruct"},
                },
            }
        ],
        "attempt_ledger": [],
        "error": None,
        "llm_calls": [
            {"model_key": "patch", "tokens": tokens},
            {"model_key": "classify", "tokens": 3},
        ],
    }


def test_run_id_for_is_stable_and_formatted():
    sweep = _load_sweep()
    rid = sweep.run_id_for("pirate chef")
    assert rid == sweep.run_id_for("pirate chef")
    assert rid.startswith("sw") and len(rid) == 6
    int(rid[2:], 16)  # 4 hex chars


def test_render_report_table_totals_and_verify_header():
    sweep = _load_sweep()
    goals = ["goal one", "goal two", "goal three"]
    artifacts = [
        _artifact(sweep.run_id_for("goal one"), "goal one"),
        _artifact(sweep.run_id_for("goal three"), "goal three", attempts=1),
    ]
    report = sweep.render_report(artifacts, goals)
    assert "# SuperTaco sweep results" in report
    assert "**How to verify:**" in report
    assert sweep.run_id_for("goal one") in report
    assert "Qwen/Qwen2.5-0.5B-Instruct" in report
    assert "nebius_job · base:real / ft:real" in report
    assert "patch:7" in report and "classify:3" in report
    assert "$" not in report  # rates empty → tokens only, no fabricated dollars
    assert "**totals**" in report
    assert "Not run" in report and "goal two" in report


def test_render_report_orders_rows_by_goals_list():
    sweep = _load_sweep()
    goals = ["a", "b"]
    shuffled = [_artifact(sweep.run_id_for("b"), "b"), _artifact(sweep.run_id_for("a"), "a")]
    report = sweep.render_report(shuffled, goals)
    assert report.index(sweep.run_id_for("a")) < report.index(sweep.run_id_for("b"))


def _wire(tmp_path, monkeypatch, sweep, goals):
    goals_path = tmp_path / "goals.json"
    goals_path.write_text(json.dumps([{"goal": g} for g in goals]), encoding="utf-8")
    monkeypatch.setattr(sweep, "GOALS_PATH", goals_path)
    monkeypatch.setattr(sweep, "ARTIFACT_DIR", tmp_path / "logs" / "runs")
    monkeypatch.setattr(sweep, "REPORT_PATH", tmp_path / "RESULTS.md")
    (tmp_path / "logs" / "runs").mkdir(parents=True)


def test_resume_skips_successful_runs_and_reports(tmp_path, monkeypatch, capsys):
    sweep = _load_sweep()
    goals = ["done goal", "pending goal"]
    _wire(tmp_path, monkeypatch, sweep, goals)
    done_id = sweep.run_id_for("done goal")
    (tmp_path / "logs" / "runs" / f"{done_id}.json").write_text(
        json.dumps(_artifact(done_id, "done goal")), encoding="utf-8"
    )
    calls = []

    class _Proc:
        returncode = 0
        stdout = "Plan written: configs/runs/plan_x.yaml (mode=fallback, dataset=synthetic)\n"
        stderr = ""

    def fake_run(argv, run_id):
        calls.append(list(argv))
        return _Proc()

    monkeypatch.setattr(sweep, "_run", fake_run)
    rc = sweep.main([])
    assert rc == 0
    plan_calls = [c for c in calls if "plan" in c]
    run_calls = [c for c in calls if "run" in c]
    assert len(plan_calls) == 1 and plan_calls[0][-1] == "pending goal"
    assert len(run_calls) == 1
    assert (tmp_path / "RESULTS.md").exists()
    timing = tmp_path / "logs" / "runs" / f"{sweep.run_id_for('pending goal')}.timing.json"
    assert timing.exists()
    assert json.loads(timing.read_text(encoding="utf-8"))["pair_s"] >= 0


def test_plans_only_never_launches_training(tmp_path, monkeypatch, capsys):
    sweep = _load_sweep()
    goals = ["g one", "g two"]
    _wire(tmp_path, monkeypatch, sweep, goals)
    calls = []

    class _Proc:
        returncode = 0
        stdout = "Plan written: configs/runs/plan_x.yaml (mode=fallback, dataset=synthetic)\n"
        stderr = ""

    def fake_run(argv, run_id):
        calls.append(list(argv))
        return _Proc()

    monkeypatch.setattr(sweep, "_run", fake_run)
    rc = sweep.main(["--plans-only"])
    assert rc == 0
    assert len(calls) == 2
    assert all("run" not in c for c in calls)
