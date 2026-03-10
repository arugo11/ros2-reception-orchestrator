from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExtractionResult:
    name: str | None = None
    affiliation: str | None = None
    purpose: str | None = None
    correction_signal: bool = False
    affirmative: bool = False


class InfoExtractor:
    """Legacy compatibility shim.

    The orchestrator now routes semantic understanding through SupervisorAdapter.
    This shim is kept only to avoid import breakage in older helper code.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        del args, kwargs

    def analyze(self, session, text: str) -> ExtractionResult:  # noqa: ANN001
        del session, text
        return ExtractionResult()
