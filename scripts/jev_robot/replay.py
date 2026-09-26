"""Safely replay one previously logged, unexecuted JEV choice.

Replay is intentionally narrow: the source run must contain no action_started
event, the current driver must still advertise the action, and the stored JEV
safety answers are evaluated again with the current guard policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .action_bridge import ActionBridge, RobotDriver
from .app import DEFAULT_RUNS, load_driver_factory
from .choices import bounded_arguments, guard_action
from .contracts import ActionOutcome, JevDecision, SceneSnapshot, SessionResult
from .events import JsonlRunRecorder
from .manifest import ActionCatalog

REPLAYABLE_ACTIONS = {"adjust_gripper"}


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_unexecuted_decision(
    source_run: Path,
    catalog: ActionCatalog,
) -> tuple[str, JevDecision]:
    records = _records(source_run / "events.jsonl")
    if any(record.get("event") == "action_started" for record in records):
        raise ValueError(
            "Source run already started an action; refusing possible duplicate execution"
        )
    choices = [record["payload"] for record in records if record.get("event") == "jev_choice"]
    if len(choices) != 1:
        raise ValueError("Replay requires exactly one logged JEV choice")
    payload = choices[0]
    answers = dict(payload.get("answers") or {})
    raw_next = dict(answers.get("next_action") or {})
    action = str(raw_next.get("choice") or "")
    if action not in REPLAYABLE_ACTIONS:
        raise ValueError(f"Action {action!r} is not approved for replay")
    catalog.require(action)
    safe_score = float((answers.get("safe_to_continue") or {}).get("noul", 0.0))
    sufficient = float((answers.get("information_sufficient") or {}).get("noul", 0.0))
    confidence = float(raw_next.get("confidence", 0.0))
    guarded = guard_action(
        action,
        confidence,
        safe_score,
        sufficient,
        {spec.name for spec in catalog.actions},
    )
    if guarded != action:
        raise ValueError(f"Current safety policy maps {action!r} to {guarded!r}")

    answer_objects = {name: SimpleNamespace(**dict(value)) for name, value in answers.items()}
    arguments = bounded_arguments(action, answer_objects)
    session = json.loads((source_run / "session.json").read_text(encoding="utf-8"))
    decision = JevDecision(
        next_action=action,
        confidence=confidence,
        probabilities={
            str(key): float(value)
            for key, value in dict(raw_next.get("probabilities") or {}).items()
        },
        arguments=arguments,
        answers=answers,
        latency_ms=float(payload.get("latency_ms") or 0.0),
    )
    return str(session["user_goal"]), decision


def _finger_width(snapshot: SceneSnapshot) -> float:
    names = [str(value) for value in snapshot.facts.get("joint_names") or []]
    positions = [float(value) for value in snapshot.facts.get("joint_positions") or []]
    values = []
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        try:
            values.append(positions[names.index(name)])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"Snapshot is missing {name}") from exc
    return sum(values) / len(values)


def replay_product_decision(
    source_run: Path,
    driver: RobotDriver,
    *,
    output_root: Path = DEFAULT_RUNS,
) -> tuple[SessionResult, Path]:
    capabilities = driver.capabilities()
    supported = capabilities.get("supported_actions")
    catalog = ActionCatalog.load()
    if supported is not None:
        catalog = catalog.only(supported)
    goal, decision = load_unexecuted_decision(source_run, catalog)
    recorder = JsonlRunRecorder(output_root, goal)
    bridge = ActionBridge(catalog, driver)
    start_session = getattr(driver, "start_session", None)
    end_session = getattr(driver, "end_session", None)
    session_started = False
    try:
        if callable(start_session):
            start_session(recorder.run_dir)
            session_started = True
        before = driver.snapshot()
        recorder.emit(
            "decision_replay_started",
            {
                "source_run": str(source_run.resolve()),
                "decision": decision.to_dict(),
                "before": before.to_dict(),
            },
        )
        recorder.emit(
            "action_started",
            {"action": decision.next_action, "arguments": decision.arguments},
        )
        raw_outcome = bridge.execute(decision)
        after = driver.snapshot()
        expected = float(raw_outcome.data["target_per_finger_m"])
        measured = _finger_width(after)
        verified = raw_outcome.success and abs(measured - expected) <= 0.0005
        outcome = ActionOutcome(
            action=raw_outcome.action,
            success=verified,
            message=(
                raw_outcome.message if verified else "Post-action finger-state verification failed"
            ),
            data={
                **raw_outcome.data,
                "measured_after_per_finger_m": measured,
                "post_action_sensor_verified": verified,
            },
        )
        recorder.emit("action_finished", outcome.to_dict())
        recorder.emit("post_action_snapshot", after.to_dict())
        result = SessionResult(
            status="completed" if verified else "failed",
            message=(
                "Stored JEV choice was revalidated, executed, and verified from robot state."
                if verified
                else "Stored JEV choice executed without matching post-action evidence."
            ),
            turns=1,
            decisions=(decision,),
            outcomes=(outcome,),
        )
        recorder.finish(result)
        return result, recorder.run_dir
    except Exception as exc:
        recorder.fail(exc)
        raise
    finally:
        if session_started and callable(end_session):
            end_session()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--driver-factory",
        default="scripts.jev_robot.drivers.isaac_rpc:create_driver",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    driver = load_driver_factory(args.driver_factory)()
    result, run_dir = replay_product_decision(
        args.source_run.resolve(),
        driver,
        output_root=args.output,
    )
    print(
        json.dumps(
            {"run_dir": str(run_dir), **result.to_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
