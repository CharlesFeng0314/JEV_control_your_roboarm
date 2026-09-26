"""Verify the physical result instead of trusting command completion."""

from backend import ManipulationBackend
from contracts import ActionResult, ObjectObservation, Pose3D


def verify_transfer(
    backend: ManipulationBackend,
    observation: ObjectObservation,
    destination: Pose3D,
) -> ActionResult:
    result = backend.verify(observation, destination)
    if not result.success:
        raise RuntimeError(result.message)
    return result
