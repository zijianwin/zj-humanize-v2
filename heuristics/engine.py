"""Replacement engine: loads YAML rules, supports exact and fuzzy matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ReplacementRule:
    """A single replacement rule loaded from YAML."""

    pattern: str
    natural: str
    balanced: str
    fuzzy: bool = False
    threshold: float = 0.85
    tags: list[str] = field(default_factory=list)
    contexts_exempt: list[str] = field(default_factory=list)
    is_regex: bool = False
    priority: int = 0  # higher = applied first

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplacementRule:
        return cls(
            pattern=str(data["pattern"]),
            natural=str(data.get("natural", data.get("replacement", ""))),
            balanced=str(data.get("balanced", data.get("natural", data.get("replacement", "")))),
            fuzzy=bool(data.get("fuzzy", False)),
            threshold=float(data.get("threshold", 0.85)),
            tags=list(data.get("tags", [])),
            contexts_exempt=list(data.get("contexts_exempt", [])),
            is_regex=bool(data.get("is_regex", False)),
            priority=int(data.get("priority", 0)),
        )


@dataclass
class ReplacementResult:
    """Result of applying replacements."""

    text: str
    applied_rules: list[str] = field(default_factory=list)
    skipped_rules: list[str] = field(default_factory=list)


class ReplacementEngine:
    """Loads rules from YAML files, applies exact or fuzzy replacements."""

    def __init__(self, rules_dir: Path | None = None):
        self.rules: list[ReplacementRule] = []
        self._rules_by_category: dict[str, list[ReplacementRule]] = {}
        if rules_dir and rules_dir.exists():
            self.load_rules_dir(rules_dir)

    def load_rules_dir(self, rules_dir: Path) -> None:
        for yaml_file in sorted(rules_dir.glob("*.yaml")):
            self.load_rules_file(yaml_file)
        for yaml_file in sorted(rules_dir.glob("*.yml")):
            self.load_rules_file(yaml_file)

    def load_rules_file(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not raw:
            return
        category = path.stem
        rules_data = raw if isinstance(raw, list) else raw.get("rules", [])
        for entry in rules_data:
            if not isinstance(entry, dict):
                continue
            rule = ReplacementRule.from_dict(entry)
            self.rules.append(rule)
            self._rules_by_category.setdefault(category, []).append(rule)
        # Sort by priority descending
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def apply(
        self,
        text: str,
        style: str = "natural",
        context: str = "",
    ) -> ReplacementResult:
        """Apply all rules. ``style`` selects 'natural' or 'balanced' replacement."""
        result_text = text
        applied: list[str] = []
        skipped: list[str] = []

        for rule in self.rules:
            # Check context exemption
            if rule.contexts_exempt and any(ctx in context for ctx in rule.contexts_exempt):
                skipped.append(rule.pattern)
                continue

            replacement = rule.natural if style == "natural" else rule.balanced
            if not replacement:
                continue

            if rule.is_regex:
                new_text = re.sub(rule.pattern, replacement, result_text)
                if new_text != result_text:
                    applied.append(rule.pattern)
                    result_text = new_text
            elif rule.fuzzy:
                new_text = self._fuzzy_replace(result_text, rule, replacement)
                if new_text != result_text:
                    applied.append(rule.pattern)
                    result_text = new_text
            else:
                if rule.pattern in result_text:
                    result_text = result_text.replace(rule.pattern, replacement)
                    applied.append(rule.pattern)

        return ReplacementResult(text=result_text, applied_rules=applied, skipped_rules=skipped)

    def _fuzzy_replace(self, text: str, rule: ReplacementRule, replacement: str) -> str:
        """Attempt fuzzy replacement using sliding window + similarity."""
        pattern_len = len(rule.pattern)
        if pattern_len < 4 or len(text) < pattern_len:
            return text

        best_ratio = 0.0
        best_start = -1
        best_end = -1

        # Sliding window: try windows of varying size around the pattern length
        for window_delta in range(-2, 5):
            window_size = pattern_len + window_delta
            if window_size < 4 or window_size > len(text):
                continue
            for start in range(0, len(text) - window_size + 1):
                segment = text[start : start + window_size]
                ratio = SequenceMatcher(None, rule.pattern, segment).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start
                    best_end = start + window_size

        if best_ratio >= rule.threshold and best_start >= 0:
            return text[:best_start] + replacement + text[best_end:]
        return text

    def get_rules_by_category(self, category: str) -> list[ReplacementRule]:
        return self._rules_by_category.get(category, [])

    @property
    def rule_count(self) -> int:
        return len(self.rules)
