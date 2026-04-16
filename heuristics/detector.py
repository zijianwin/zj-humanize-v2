"""Template phrase detector with severity levels and context awareness."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class PhraseSeverity(Enum):
    """How strongly a phrase signals AI-generated text."""

    CRITICAL = "critical"      # 极强 AI 味, heavy penalty
    HIGH = "high"              # 明显 AI 味
    MODERATE = "moderate"      # 有模板感, 但真人也会用
    LOW = "low"                # 行业术语, 某些场景下正常


SEVERITY_PENALTY = {
    PhraseSeverity.CRITICAL: 0.25,
    PhraseSeverity.HIGH: 0.16,
    PhraseSeverity.MODERATE: 0.08,
    PhraseSeverity.LOW: 0.04,
}


@dataclass
class TemplatePhrase:
    """A single template phrase with metadata."""

    phrase: str
    severity: PhraseSeverity = PhraseSeverity.HIGH
    category: str = ""
    contexts_exempt: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplatePhrase:
        severity_str = str(data.get("severity", "high")).lower()
        try:
            severity = PhraseSeverity(severity_str)
        except ValueError:
            severity = PhraseSeverity.HIGH
        return cls(
            phrase=str(data["phrase"]),
            severity=severity,
            category=str(data.get("category", "")),
            contexts_exempt=list(data.get("contexts_exempt", [])),
        )


@dataclass
class DetectionResult:
    """Result of scanning text for template phrases."""

    hits: list[TemplatePhrase]
    total_penalty: float
    details: list[dict[str, Any]]

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def has_critical(self) -> bool:
        return any(h.severity == PhraseSeverity.CRITICAL for h in self.hits)


class TemplatePhraseDetector:
    """Detects template phrases in text with severity-aware penalties."""

    def __init__(self, phrases_path: Path | None = None):
        self.phrases: list[TemplatePhrase] = []
        if phrases_path and phrases_path.exists():
            self.load(phrases_path)

    def load(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not raw:
            return
        items = raw if isinstance(raw, list) else raw.get("phrases", [])
        for entry in items:
            if isinstance(entry, dict):
                self.phrases.append(TemplatePhrase.from_dict(entry))
            elif isinstance(entry, str):
                self.phrases.append(TemplatePhrase(phrase=entry))

    def add_phrase(self, phrase: str, severity: PhraseSeverity = PhraseSeverity.HIGH, **kwargs: Any) -> None:
        self.phrases.append(TemplatePhrase(phrase=phrase, severity=severity, **kwargs))

    def detect(self, text: str, context: str = "") -> DetectionResult:
        """Scan text for template phrases, respecting context exemptions."""
        hits: list[TemplatePhrase] = []
        details: list[dict[str, Any]] = []
        total_penalty = 0.0

        for phrase_obj in self.phrases:
            if phrase_obj.phrase not in text:
                continue
            # Check exemption
            if phrase_obj.contexts_exempt and any(ctx in context for ctx in phrase_obj.contexts_exempt):
                details.append({
                    "phrase": phrase_obj.phrase,
                    "severity": phrase_obj.severity.value,
                    "status": "exempt",
                    "penalty": 0.0,
                })
                continue

            penalty = SEVERITY_PENALTY.get(phrase_obj.severity, 0.10)
            total_penalty += penalty
            hits.append(phrase_obj)
            details.append({
                "phrase": phrase_obj.phrase,
                "severity": phrase_obj.severity.value,
                "category": phrase_obj.category,
                "status": "hit",
                "penalty": penalty,
            })

        return DetectionResult(
            hits=hits,
            total_penalty=min(1.0, total_penalty),
            details=details,
        )

    def source_reduction_score(self, source_text: str, candidate: str, context: str = "") -> float:
        """Check how many source template phrases were removed in the candidate."""
        source_hits = [p for p in self.phrases if p.phrase in source_text]
        if not source_hits:
            return 1.0
        carried = [p for p in source_hits if p.phrase in candidate]
        if not carried:
            return 1.0
        # Weight by severity
        carried_weight = sum(SEVERITY_PENALTY.get(p.severity, 0.10) for p in carried)
        source_weight = sum(SEVERITY_PENALTY.get(p.severity, 0.10) for p in source_hits)
        ratio = carried_weight / max(source_weight, 0.01)
        return max(0.0, 1.0 - ratio)

    @property
    def phrase_count(self) -> int:
        return len(self.phrases)
