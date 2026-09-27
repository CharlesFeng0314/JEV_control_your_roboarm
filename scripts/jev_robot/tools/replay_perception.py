"""Replay product wrist perception from a recorded JEV run without Isaac Sim."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.jev_robot.drivers.isaac_rpc import IsaacRpcDriver
from scripts.manipulation.perception.wrist_semantics import WristRgbdSemanticPerception

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "scripts" / "manipulation" / "perception" / "wrist_rgbd.json"


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def recorded_packets(run_dir: Path) -> list[dict[str, Any]]:
    sidecars = sorted((run_dir / "artifacts" / "wrist_rgbd").glob("*_packet.json"))
    if sidecars:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sidecars]
    packets: dict[str, dict[str, Any]] = {}
    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            for item in _walk(json.loads(line)):
                wrist = item.get("wrist_rgbd")
                if not isinstance(wrist, dict) or not wrist.get("rgb_path"):
                    continue
                packets[str(wrist["rgb_path"])] = {
                    "rgb_path": wrist["rgb_path"],
                    "depth_path": wrist["depth_path"],
                    "intrinsics": wrist["intrinsics"],
                    "pose": wrist["camera_pose"],
                    "orientation_wxyz": wrist["camera_orientation_wxyz"],
                }
    return list(packets.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    perception = WristRgbdSemanticPerception(args.config.resolve(), PROJECT_ROOT)
    driver = IsaacRpcDriver(perception=perception, transport=lambda _payload, _timeout: {})
    records = []
    try:
        for sequence, packet in enumerate(recorded_packets(args.run_dir.resolve()), start=1):
            started = time.perf_counter()
            observations = perception.detect(packet)
            driver._sensor_sequence = sequence
            tracked = driver._track_observations(observations)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record = {
                "frame": Path(packet["rgb_path"]).name,
                "latency_ms": elapsed_ms,
                "objects": [
                    {
                        "object_id": item.get("object_id"),
                        "label": item.get("label"),
                        "confidence": item.get("confidence"),
                        "detector_label": item.get("detector_label"),
                        "detector_score": item.get("detector_score"),
                        "semantic_scores": item.get("semantic_scores"),
                        "position_m": (item.get("pose") or {}).get("position_m"),
                    }
                    for item in tracked
                ],
            }
            records.append(record)
            labels = ", ".join(
                f"{item['object_id']}={item['label']}:{item['confidence']:.3f}"
                for item in record["objects"]
            )
            print(f"{record['frame']} {elapsed_ms:.1f} ms {labels or 'no objects'}", flush=True)
    finally:
        perception.close()

    result = {
        "schema_version": 1,
        "source_run": str(args.run_dir.resolve()),
        "method": "rgbd_world_cluster_semantic_fusion",
        "device": "cuda:0",
        "frames": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
