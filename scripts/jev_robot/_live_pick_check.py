"""One-shot live Isaac validation for the product wrist-RGB-D pick path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scripts.jev_robot.drivers.isaac_rpc import (
    ARM_JOINTS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    SEARCH_ARM_POSES,
    IsaacRpcDriver,
    _socket_rpc,
)
from scripts.jev_robot.manifest import ActionSpec


def action(name: str) -> ActionSpec:
    return ActionSpec(name, name, f"actions/{name}.py", {}, {})


def main() -> None:
    run_dir = Path("data/jev_robot/live_pick_checks") / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict] = []
    last_phase = None

    def traced_transport(payload: dict, timeout: float) -> dict:
        nonlocal last_phase
        result = _socket_rpc(DEFAULT_HOST, DEFAULT_PORT, payload, timeout)
        if payload.get("cmd") == "set_pick":
            print("WAYPOINT=" + json.dumps(payload["position"]), flush=True)
        if payload.get("cmd") == "manipulation_step":
            item = {"robot": result, "validation_object": _socket_rpc(
                DEFAULT_HOST, DEFAULT_PORT, {"cmd": "object_state"}, timeout
            )}
            trace.append(item)
            if result.get("phase") != last_phase:
                last_phase = result.get("phase")
                print("PHASE=" + json.dumps(item), flush=True)
        return result

    driver = IsaacRpcDriver(transport=traced_transport)
    driver.start_session(run_dir)
    try:
        driver._call({"cmd": "reset_validation_scene", "settle_steps": 120}, timeout=180.0)
        before = driver._call({"cmd": "object_state"})
        pose = SEARCH_ARM_POSES[0]
        driver._call(
            {
                "cmd": "set_joints",
                "targets": dict(zip(ARM_JOINTS, pose, strict=True)),
                "steps": 220,
                "tolerance": 0.06,
            },
            timeout=180.0,
        )
        observed = driver.execute_action(
            action("observe_object"),
            {"target_label": "sugar_box", "search_scope": "wide"},
        )
        (run_dir / "observation.json").write_text(
            json.dumps(observed.to_dict(), indent=2), encoding="utf-8"
        )
        print("OBSERVE=" + json.dumps(observed.data.get("observation")), flush=True)
        if not observed.success or not observed.data.get("observation"):
            raise RuntimeError(observed.message)

        target_ref = observed.data["observation"]["object_id"]
        driver._call({"cmd": "set_overlay", "action": "pick_object"})
        picked = driver.execute_action(action("pick_object"), {"target_ref": target_ref})
        print("PICK=" + json.dumps(picked.to_dict(), ensure_ascii=True), flush=True)
        diagnostics = driver._call({"cmd": "validation_diagnostics"})
        (run_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        after = driver._call({"cmd": "object_state"})
        if picked.success:
            driver._call({"cmd": "step", "n": 120})
        held = driver._call({"cmd": "object_state"})
        measured_lift = float(after["position_m"][2] - before["position_m"][2])
        held_lift = float(held["position_m"][2] - before["position_m"][2])
        validation = {
            "before": before, "after": after, "held_after_120_steps": held,
            "measured_lift_m": measured_lift, "held_lift_m": held_lift,
        }
        print("VALIDATION=" + json.dumps(validation), flush=True)
        (run_dir / "result.json").write_text(
            json.dumps({"pick": picked.to_dict(), **validation}, indent=2),
            encoding="utf-8",
        )
        if not picked.success or not picked.data.get("hold_verified"):
            raise RuntimeError(picked.message)
        if measured_lift < 0.04 or held_lift < 0.04:
            raise RuntimeError("Validation did not confirm the sugar box lifted and stayed held")
    finally:
        (run_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        driver.end_session()


if __name__ == "__main__":
    main()
