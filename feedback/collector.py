"""User feedback collection and persistence.

Collects post-run feedback (thumbs up/down, free-text, manual edits) and
feeds it back into the strategy state and style learning systems.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FeedbackRecord:
    """One piece of user feedback for a humanize run."""

    task: str
    scenario: str
    source_text: str
    output_text: str
    final_score: float
    rating: str = ""                # "good" | "bad" | "neutral" | ""
    comment: str = ""               # free-text feedback
    manual_edit: str = ""           # user's manually corrected version
    failure_areas: list[str] = field(default_factory=list)
    timestamp: str = ""
    winning_profile: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "scenario": self.scenario,
            "source_text": self.source_text[:300],
            "output_text": self.output_text[:300],
            "final_score": round(self.final_score, 4),
            "rating": self.rating,
            "comment": self.comment,
            "manual_edit": self.manual_edit[:300] if self.manual_edit else "",
            "failure_areas": self.failure_areas,
            "timestamp": self.timestamp,
            "winning_profile": self.winning_profile,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        return cls(
            task=str(data.get("task", "")),
            scenario=str(data.get("scenario", "")),
            source_text=str(data.get("source_text", "")),
            output_text=str(data.get("output_text", "")),
            final_score=float(data.get("final_score", 0)),
            rating=str(data.get("rating", "")),
            comment=str(data.get("comment", "")),
            manual_edit=str(data.get("manual_edit", "")),
            failure_areas=list(data.get("failure_areas", [])),
            timestamp=str(data.get("timestamp", "")),
            winning_profile=str(data.get("winning_profile", "")),
        )


class FeedbackStore:
    """Persists and queries user feedback."""

    def __init__(self, feedback_path: Path | None = None):
        self.feedback_path = feedback_path
        self.records: list[FeedbackRecord] = []
        if feedback_path and feedback_path.exists():
            self._load()

    def _load(self) -> None:
        if not self.feedback_path:
            return
        try:
            data = json.loads(self.feedback_path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("records", [])
            for item in items:
                self.records.append(FeedbackRecord.from_dict(item))
        except (json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        if not self.feedback_path:
            return
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": [r.as_dict() for r in self.records[-500:]],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.feedback_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, record: FeedbackRecord) -> None:
        self.records.append(record)
        self.save()

    def get_bad_examples(self, limit: int = 20) -> list[FeedbackRecord]:
        """Return recent negative-feedback examples for learning."""
        return [r for r in reversed(self.records) if r.rating == "bad"][:limit]

    def get_good_examples(self, limit: int = 20) -> list[FeedbackRecord]:
        """Return recent positive-feedback examples for learning."""
        return [r for r in reversed(self.records) if r.rating == "good"][:limit]

    def get_manual_edits(self, limit: int = 20) -> list[FeedbackRecord]:
        """Return records where user provided manual corrections."""
        return [r for r in reversed(self.records) if r.manual_edit.strip()][:limit]

    def scenario_summary(self, scenario: str) -> dict[str, Any]:
        """Aggregate feedback stats for a scenario."""
        relevant = [r for r in self.records if r.scenario == scenario]
        if not relevant:
            return {"count": 0}
        good = sum(1 for r in relevant if r.rating == "good")
        bad = sum(1 for r in relevant if r.rating == "bad")
        avg_score = sum(r.final_score for r in relevant) / len(relevant)
        return {
            "count": len(relevant),
            "good": good,
            "bad": bad,
            "neutral": len(relevant) - good - bad,
            "avg_score": round(avg_score, 4),
            "satisfaction_rate": round(good / len(relevant), 4) if relevant else 0,
        }

    @property
    def total_count(self) -> int:
        return len(self.records)
