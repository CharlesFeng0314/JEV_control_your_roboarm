"""Robot-mounted perception used by both simulation and real-driver adapters."""

from .wrist_semantics import LABEL_CRITERIA, WristRgbdSemanticPerception

__all__ = ["LABEL_CRITERIA", "WristRgbdSemanticPerception"]
