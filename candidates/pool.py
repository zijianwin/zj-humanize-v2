"""Candidate pool: manages multiple candidates, picks best."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    """One candidate in the pool."""

    index: int
    profile: str
    source_kind: str  # "model_direct" | "model_repair" | "heuristic" | "manual"
    text: str
    score: dict[str, Any] | None = None
    failure_tags: list[str] = field(default_factory=list)
    applied_rules: list[str] = field(default_factory=list)
    generation_response: dict[str, Any] | None = None
    error: str = ""

    def rank_key(self) -> tuple[int, float, float]:
        s = self.score or {}
        return (
            1 if not s.get("hard_fail") else 0,
            float(s.get("final_score") or 0.0),
            float(s.get("model_score") or 0.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.index,
            "profile": self.profile,
            "source_kind": self.source_kind,
            "text": self.text,
            "score": self.score,
            "failure_tags": self.failure_tags,
            "applied_rules": self.applied_rules,
            "error": self.error,
        }


class CandidatePool:
    """Collects candidates and picks the best."""

    def __init__(self) -> None:
        self.candidates: list[CandidateRecord] = []
        self._next_index = 1

    def add(
        self,
        profile: str,
        source_kind: str,
        text: str,
        score: dict[str, Any] | None = None,
        failure_tags: list[str] | None = None,
        applied_rules: list[str] | None = None,
        generation_response: dict[str, Any] | None = None,
        error: str = "",
    ) -> CandidateRecord:
        record = CandidateRecord(
            index=self._next_index,
            profile=profile,
            source_kind=source_kind,
            text=text,
            score=score,
            failure_tags=failure_tags or [],
            applied_rules=applied_rules or [],
            generation_response=generation_response,
            error=error,
        )
        self.candidates.append(record)
        self._next_index += 1
        return record

    def pick_best(self) -> CandidateRecord:
        valid = [c for c in self.candidates if c.score and not c.score.get("hard_fail")]
        pool = valid or [c for c in self.candidates if c.score]
        if not pool:
            raise RuntimeError("No candidate produced a scorable output")
        return max(pool, key=lambda c: c.rank_key())

    @property
    def count(self) -> int:
        return len(self.candidates)

    def all_dicts(self) -> list[dict[str, Any]]:
        return [c.as_dict() for c in self.candidates]
