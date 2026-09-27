"""Shared operator-facing helpers for the local web product UI."""

from __future__ import annotations


def humanize_service_error(exc: Exception) -> str:
    """Return an operator-facing error without dumping SDK internals or credentials."""

    message = str(exc)
    lowered = message.lower()
    if "401" in message or "authenticate" in lowered or "authentication" in lowered:
        return (
            "TypeSafe authentication failed (HTTP 401). No robot action was executed. "
            "Create a new official API secret, set TYPESAFE_API_KEY in .env, and retry."
        )
    if "429" in message or "high demand" in lowered:
        return (
            "JEV service is busy (HTTP 429). No unfinished robot action was "
            "continued; retry or resume the recorded run shortly."
        )
    return message
