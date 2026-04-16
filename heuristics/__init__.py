"""Heuristic replacement engine for humanizing Chinese text."""

from heuristics.engine import ReplacementEngine, ReplacementRule
from heuristics.detector import TemplatePhraseDetector, PhraseSeverity

__all__ = [
    "ReplacementEngine",
    "ReplacementRule",
    "TemplatePhraseDetector",
    "PhraseSeverity",
]
