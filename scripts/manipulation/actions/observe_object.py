"""Observe and localize one target object before any motion is authorized."""

from backend import ManipulationBackend
from contracts import ObjectObservation


def _accept(observation: ObjectObservation, target_label: str) -> ObjectObservation:
    if observation.label != target_label:
        raise RuntimeError(
            f"Requested {target_label!r}, but perception returned {observation.label!r}"
        )
    if observation.confidence <= 0.0:
        raise RuntimeError("Perception confidence must be positive")
    return observation


def observe_object(backend: ManipulationBackend, target_label: str) -> ObjectObservation:
    if hasattr(backend, "perceive") and hasattr(backend, "aim_gripper_for_search"):
        from .search_object import search_object

        return search_object(backend, target_label)
    return _accept(backend.observe(target_label), target_label)
