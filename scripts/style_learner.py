"""Style learning from user examples.

Analyzes user-provided writing samples (or manual edits from feedback) to
extract common stylistic patterns: sentence length, punctuation usage,
paragraph structure, vocabulary preferences. These patterns can then inform
scoring weights and heuristic selection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StyleProfile:
    """Learned style characteristics from example texts."""

    name: str = "default"
    avg_sentence_length: float = 0.0     # chars per sentence
    avg_paragraph_length: float = 0.0    # sentences per paragraph
    exclamation_rate: float = 0.0        # ratio of sentences ending with !
    question_rate: float = 0.0           # ratio of sentences ending with ?
    emoji_rate: float = 0.0             # emojis per 100 chars
    preferred_openers: list[str] = field(default_factory=list)
    preferred_closers: list[str] = field(default_factory=list)
    vocab_preferences: dict[str, str] = field(default_factory=dict)  # avoid -> prefer
    sample_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "avg_paragraph_length": round(self.avg_paragraph_length, 2),
            "exclamation_rate": round(self.exclamation_rate, 4),
            "question_rate": round(self.question_rate, 4),
            "emoji_rate": round(self.emoji_rate, 4),
            "preferred_openers": self.preferred_openers[:10],
            "preferred_closers": self.preferred_closers[:10],
            "vocab_preferences": dict(list(self.vocab_preferences.items())[:30]),
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleProfile:
        return cls(
            name=str(data.get("name", "default")),
            avg_sentence_length=float(data.get("avg_sentence_length", 0)),
            avg_paragraph_length=float(data.get("avg_paragraph_length", 0)),
            exclamation_rate=float(data.get("exclamation_rate", 0)),
            question_rate=float(data.get("question_rate", 0)),
            emoji_rate=float(data.get("emoji_rate", 0)),
            preferred_openers=list(data.get("preferred_openers", [])),
            preferred_closers=list(data.get("preferred_closers", [])),
            vocab_preferences=dict(data.get("vocab_preferences", {})),
            sample_count=int(data.get("sample_count", 0)),
        )


class StyleLearner:
    """Learns stylistic patterns from example texts."""

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"
        "\U0000fe00-\U0000fe0f"
        "\U0001f900-\U0001f9ff"
        "]+",
        flags=re.UNICODE,
    )

    SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+")

    def __init__(self, profile_path: Path | None = None):
        self.profile_path = profile_path
        self.profile = StyleProfile()
        if profile_path and profile_path.exists():
            self._load()

    def _load(self) -> None:
        if not self.profile_path:
            return
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            self.profile = StyleProfile.from_dict(data)
        except (json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        if not self.profile_path:
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(self.profile.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def learn_from_text(self, text: str) -> StyleProfile:
        """Analyze a single text and update the running profile."""
        analysis = self._analyze(text)
        self._merge(analysis)
        self.profile.sample_count += 1
        return self.profile

    def learn_from_texts(self, texts: list[str]) -> StyleProfile:
        """Analyze multiple texts and build a composite profile."""
        for text in texts:
            self.learn_from_text(text)
        return self.profile

    def learn_from_edit_pair(self, original: str, edited: str) -> None:
        """Learn vocabulary preferences from before/after editing pairs.

        Detects phrases removed in the edit and their replacements.
        """
        # Simple diff-based learning: find removed and added short phrases
        orig_sentences = self._split_sentences(original)
        edit_sentences = self._split_sentences(edited)

        # Learn from the edited version's style
        self.learn_from_text(edited)

        # Extract vocabulary shifts
        orig_set = set(self._extract_phrases(original))
        edit_set = set(self._extract_phrases(edited))
        removed = orig_set - edit_set
        added = edit_set - orig_set

        # Simple heuristic: if a removed phrase and an added phrase share
        # the same position context, treat it as a preference
        for rem in list(removed)[:10]:
            for add in list(added)[:10]:
                if len(rem) >= 2 and len(add) >= 2:
                    self.profile.vocab_preferences[rem] = add

    def get_style_hints(self) -> dict[str, Any]:
        """Return style hints that can be injected into generation prompts."""
        hints: dict[str, Any] = {}
        p = self.profile

        if p.sample_count < 1:
            return hints

        if p.avg_sentence_length > 0:
            hints["target_sentence_length"] = round(p.avg_sentence_length)

        if p.exclamation_rate > 0.3:
            hints["tone"] = "enthusiastic"
        elif p.exclamation_rate < 0.05:
            hints["tone"] = "calm"

        if p.emoji_rate > 0.5:
            hints["use_emoji"] = True

        if p.preferred_openers:
            hints["opener_examples"] = p.preferred_openers[:3]

        if p.preferred_closers:
            hints["closer_examples"] = p.preferred_closers[:3]

        if p.vocab_preferences:
            hints["vocab_preferences"] = dict(list(p.vocab_preferences.items())[:10])

        return hints

    # ----- internal analysis -----

    def _analyze(self, text: str) -> dict[str, Any]:
        sentences = self._split_sentences(text)
        if not sentences:
            return {}

        lengths = [len(re.sub(r"\s+", "", s)) for s in sentences]
        excl = sum(1 for s in sentences if s.strip().endswith(("!", "！")))
        ques = sum(1 for s in sentences if s.strip().endswith(("?", "？")))
        emojis = self.EMOJI_PATTERN.findall(text)
        char_count = len(re.sub(r"\s+", "", text))

        # Extract openers and closers
        opener = sentences[0].strip()[:20] if sentences else ""
        closer = sentences[-1].strip()[-20:] if len(sentences) > 1 else ""

        # Paragraph analysis
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sents_per_para = [
            len([s for s in self.SENTENCE_SPLIT.split(p) if s.strip()])
            for p in paragraphs
        ] or [1]

        return {
            "avg_sentence_length": sum(lengths) / len(lengths) if lengths else 0,
            "avg_paragraph_length": sum(sents_per_para) / len(sents_per_para),
            "exclamation_rate": excl / len(sentences),
            "question_rate": ques / len(sentences),
            "emoji_rate": (len(emojis) / max(char_count, 1)) * 100,
            "opener": opener,
            "closer": closer,
        }

    def _merge(self, analysis: dict[str, Any]) -> None:
        """Merge a single analysis into the running profile using moving average."""
        if not analysis:
            return
        n = self.profile.sample_count
        weight_old = n / (n + 1) if n > 0 else 0
        weight_new = 1 / (n + 1) if n > 0 else 1

        self.profile.avg_sentence_length = (
            self.profile.avg_sentence_length * weight_old
            + analysis.get("avg_sentence_length", 0) * weight_new
        )
        self.profile.avg_paragraph_length = (
            self.profile.avg_paragraph_length * weight_old
            + analysis.get("avg_paragraph_length", 0) * weight_new
        )
        self.profile.exclamation_rate = (
            self.profile.exclamation_rate * weight_old
            + analysis.get("exclamation_rate", 0) * weight_new
        )
        self.profile.question_rate = (
            self.profile.question_rate * weight_old
            + analysis.get("question_rate", 0) * weight_new
        )
        self.profile.emoji_rate = (
            self.profile.emoji_rate * weight_old
            + analysis.get("emoji_rate", 0) * weight_new
        )

        opener = analysis.get("opener", "")
        if opener and opener not in self.profile.preferred_openers:
            self.profile.preferred_openers.append(opener)
            self.profile.preferred_openers = self.profile.preferred_openers[-15:]

        closer = analysis.get("closer", "")
        if closer and closer not in self.profile.preferred_closers:
            self.profile.preferred_closers.append(closer)
            self.profile.preferred_closers = self.profile.preferred_closers[-15:]

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self.SENTENCE_SPLIT.split(text) if s.strip()]

    def _extract_phrases(self, text: str) -> list[str]:
        """Extract 2-4 char phrases for vocabulary comparison."""
        c = re.sub(r"\s+", "", text)
        phrases = []
        for length in (2, 3, 4):
            for i in range(len(c) - length + 1):
                phrases.append(c[i : i + length])
        return phrases
