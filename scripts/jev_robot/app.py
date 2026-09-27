"""CLI and local web entry point for the modular JEV robot product."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path

from .action_bridge import ActionBridge, RobotDriver
from .choices import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    JevDecisionEngine,
    load_api_settings,
)
from .contracts import SessionResult
from .events import CompositeEventSink, EventSink, JsonlRunRecorder
from .manifest import ActionCatalog
from .orchestrator import ProductOrchestrator
from .recovery import ResumeContext, load_resume_context
from .scene_memory import SceneMemory
from .ui import humanize_service_error
from .web_ui import run_web_ui

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = PROJECT_ROOT / ".env"
DEFAULT_RUNS = PROJECT_ROOT / "data" / "jev_robot" / "runs"
DEFAULT_SCENES = PROJECT_ROOT / "data" / "jev_robot" / "scenes"
DriverFactory = Callable[[], RobotDriver]


def load_wrist_frame_source(reference: str) -> Callable[[], dict] | None:
    """Load an optional lightweight camera source from the selected driver module."""

    module_name, _attribute = reference.split(":", 1)
    source_factory = getattr(
        importlib.import_module(module_name),
        "create_wrist_frame_source",
        None,
    )
    return source_factory() if callable(source_factory) else None


def load_driver_factory(reference: str) -> DriverFactory:
    try:
        module_name, attribute = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError("Driver factory must use module:function syntax") from exc
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Driver factory {reference!r} is not callable")
    return factory


def run_product_session(
    goal: str,
    driver_factory: DriverFactory,
    *,
    api_key: str,
    api_model: str = DEFAULT_MODEL,
    api_base_url: str = DEFAULT_BASE_URL,
    api_provider: str = "typesafe_official",
    output_root: Path = DEFAULT_RUNS,
    scene_id: str = "active_workspace",
    memory_root: Path = DEFAULT_SCENES,
    ui_sink: EventSink | None = None,
    resume_context: ResumeContext | None = None,
) -> tuple[SessionResult, Path]:
    recorder = JsonlRunRecorder(
        output_root,
        goal,
        scene_id=scene_id,
        parent_run=resume_context.source_run if resume_context else None,
    )
    sink: EventSink = recorder
    if ui_sink is not None:
        sink = CompositeEventSink(recorder, ui_sink)
    driver = driver_factory()
    recorder.emit(
        "jev_configuration",
        {
            "provider": api_provider,
            "base_url": api_base_url,
            "model": api_model,
        },
    )
    capabilities = driver.capabilities()
    supported_actions = capabilities.get("supported_actions")
    catalog = ActionCatalog.load()
    if supported_actions is not None:
        catalog = catalog.only(supported_actions)
    memory = SceneMemory.open(memory_root, scene_id)
    orchestrator = ProductOrchestrator(
        catalog,
        JevDecisionEngine(api_key, model=api_model, base_url=api_base_url),
        ActionBridge(catalog, driver),
        sink=sink,
        memory=memory,
    )
    start_session = getattr(driver, "start_session", None)
    end_session = getattr(driver, "end_session", None)
    session_started = False
    try:
        if callable(start_session):
            start_session(recorder.run_dir)
            session_started = True
        result = orchestrator.run(
            goal,
            initial_decisions=resume_context.decisions if resume_context else (),
            initial_outcomes=resume_context.outcomes if resume_context else (),
            post_action_observation_seen=(
                resume_context.post_action_observation_seen if resume_context else False
            ),
        )
    except Exception as exc:
        recorder.fail(exc)
        raise
    finally:
        if session_started and callable(end_session):
            end_session()
    recorder.finish(result)
    return result, recorder.run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driver-factory",
        required=True,
        help="Import path module:function returning a sim or real RobotDriver.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--goal", help="Run headlessly; omit to open the local web UI.")
    mode.add_argument("--resume-run", type=Path, help="Continue a safely checkpointed failed run.")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--scene-id")
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = load_api_settings(args.env)
    factory = load_driver_factory(args.driver_factory)
    wrist_frame_source = load_wrist_frame_source(args.driver_factory)
    resume_context = load_resume_context(args.resume_run) if args.resume_run else None
    goal = args.goal or (resume_context.user_goal if resume_context else None)
    scene_id = args.scene_id or (
        resume_context.scene_id
        if resume_context and resume_context.scene_id
        else "active_workspace"
    )
    if goal:
        try:
            result, run_dir = run_product_session(
                goal,
                factory,
                api_key=api.api_key,
                api_model=api.model,
                api_base_url=api.base_url,
                api_provider=api.provider,
                output_root=args.output,
                scene_id=scene_id,
                memory_root=args.memory_root,
                resume_context=resume_context,
            )
        except Exception as exc:
            message = humanize_service_error(exc)
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": message,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 3
        print(
            json.dumps({"run_dir": str(run_dir), **result.to_dict()}, ensure_ascii=False, indent=2)
        )
        return 0 if result.completed else 2

    def gui_session(goal: str, sink: EventSink) -> SessionResult:
        result, _run_dir = run_product_session(
            goal,
            factory,
            api_key=api.api_key,
            api_model=api.model,
            api_base_url=api.base_url,
            api_provider=api.provider,
            output_root=args.output,
            ui_sink=sink,
            scene_id=scene_id,
            memory_root=args.memory_root,
        )
        return result

    run_web_ui(
        gui_session,
        host=args.web_host,
        port=args.web_port,
        open_browser=not args.no_browser,
        wrist_frame_source=wrist_frame_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
