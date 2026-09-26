"""Place a held object at a structured pose."""

from backend import ManipulationBackend
from contracts import ActionResult, Pose3D


def place_object(backend: ManipulationBackend, destination: Pose3D) -> ActionResult:
    result = backend.place(destination)
    if not result.success:
        raise RuntimeError(result.message)
    return result
