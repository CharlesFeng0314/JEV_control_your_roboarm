"""Move the gripper camera until the target is in view, then localize it."""

from contracts import ObjectObservation

SEARCH_SCOPE_FRACTIONS = {
    "current_view": 0.0,
    "narrow": 0.5,
    "wide": 1.0,
}


def search_object(
    backend,
    target_label: str,
    search_scope: str = "wide",
) -> ObjectObservation:
    """Look once, then use a JEV-selected fraction of the available search views."""
    if search_scope not in SEARCH_SCOPE_FRACTIONS:
        raise ValueError(f"Unknown search scope {search_scope!r}")
    if hasattr(backend, "prepare_search"):
        backend.prepare_search()
    found = backend.perceive(target_label)
    if found is None:
        print("当前视角没有找到目标，开始移动腕部相机寻找。", flush=True)
        available_views = int(backend.search_view_count())
        fraction = SEARCH_SCOPE_FRACTIONS[search_scope]
        view_count = min(available_views, int(available_views * fraction + 0.999))
        for index in range(view_count):
            backend.aim_gripper_for_search(index)
            found = backend.perceive(target_label)
            if found is not None:
                print(f"视角 {index + 1} 找到 {target_label}，开始抓取。", flush=True)
                break
    if found is None:
        message = "Geometry-first RGB-D clustering found no valid tabletop objects"
        if hasattr(backend, "search_failure_message"):
            message = backend.search_failure_message()
        raise RuntimeError(message)
    if found.label != target_label:
        raise RuntimeError(f"Requested {target_label!r}, but perception returned {found.label!r}")
    if found.confidence <= 0.0:
        raise RuntimeError("Perception confidence must be positive")
    if hasattr(backend, "finish_observation"):
        backend.finish_observation(found)
    return found
