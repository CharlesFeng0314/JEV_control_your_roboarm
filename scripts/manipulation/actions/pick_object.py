"""Pick a previously observed object; never accepts free-form motor commands."""

from backend import ManipulationBackend
from contracts import ActionResult, ObjectObservation


def pick_object(backend: ManipulationBackend, observation: ObjectObservation) -> ActionResult:
    if hasattr(backend, "begin_pick") and hasattr(backend, "advance_pick"):
        backend.begin_pick(observation)
        result = None
        while result is None:
            if hasattr(backend, "correct_approach_from_wrist"):
                backend.correct_approach_from_wrist()
            result = backend.advance_pick()
    else:
        result = backend.pick(observation)
    if not result.success:
        raise RuntimeError(result.message)
    return result
