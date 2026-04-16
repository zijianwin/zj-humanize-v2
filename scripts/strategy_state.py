"""Self-evolving strategy state machine.

Tracks profile performance, replacement rule effectiveness, and adapts
the generation strategy over time. Persists state to JSON for cross-session
learning.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Profile statistics
# ---------------------------------------------------------------------------

@dataclass
class ProfileStats:
    """Win/loss statistics for a generation profile."""

    wins: int = 0
    losses: int = 0
    total_score: float = 0.0
    uses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def avg_score(self) -> float:
        return self.total_score / self.uses if self.uses > 0 else 0.0

    def record(self, won: bool, score: float) -> None:
        self.uses += 1
        self.total_score += score
        if won:
            self.wins += 1
        else:
            self.losses += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "total_score": round(self.total_score, 4),
            "uses": self.uses,
            "win_rate": round(self.win_rate, 4),
            "avg_score": round(self.avg_score, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileStats:
        return cls(
            wins=int(data.get("wins", 0)),
            losses=int(data.get("losses", 0)),
            total_score=float(data.get("total_score", 0)),
            uses=int(data.get("uses", 0)),
        )


# ---------------------------------------------------------------------------
# Rule effectiveness tracking
# ---------------------------------------------------------------------------

@dataclass
class RuleStats:
    """Effectiveness stats for a single replacement rule."""

    applied_count: int = 0
    improved_count: int = 0       # times applying this rule led to score improvement
    worsened_count: int = 0       # times applying this rule led to score decrease
    total_delta: float = 0.0      # cumulative score change

    @property
    def effectiveness(self) -> float:
        """Positive = generally helpful, negative = generally harmful."""
        if self.applied_count == 0:
            return 0.0
        return self.total_delta / self.applied_count

    @property
    def improvement_rate(self) -> float:
        if self.applied_count == 0:
            return 0.5
        return self.improved_count / self.applied_count

    def record(self, delta: float) -> None:
        self.applied_count += 1
        self.total_delta += delta
        if delta > 0.005:
            self.improved_count += 1
        elif delta < -0.005:
            self.worsened_count += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied_count": self.applied_count,
            "improved_count": self.improved_count,
            "worsened_count": self.worsened_count,
            "total_delta": round(self.total_delta, 6),
            "effectiveness": round(self.effectiveness, 6),
            "improvement_rate": round(self.improvement_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleStats:
        return cls(
            applied_count=int(data.get("applied_count", 0)),
            improved_count=int(data.get("improved_count", 0)),
            worsened_count=int(data.get("worsened_count", 0)),
            total_delta=float(data.get("total_delta", 0)),
        )


# ---------------------------------------------------------------------------
# Strategy State
# ---------------------------------------------------------------------------

class StrategyState:
    """Persistent strategy state tracking profiles and rules."""

    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path
        self.profiles: dict[str, ProfileStats] = {}
        self.rules: dict[str, RuleStats] = {}
        self.scenario_history: list[dict[str, Any]] = []
        self._policy_overrides: dict[str, Any] = {}
        self._loaded = False

        if state_path and state_path.exists():
            self._load()

    # ----- persistence -----

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name, pdata in data.get("profiles", {}).items():
            self.profiles[name] = ProfileStats.from_dict(pdata)
        for name, rdata in data.get("rules", {}).items():
            self.rules[name] = RuleStats.from_dict(rdata)
        self.scenario_history = data.get("scenario_history", [])
        self._policy_overrides = data.get("policy_overrides", {})
        self._loaded = True

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profiles": {n: p.as_dict() for n, p in self.profiles.items()},
            "rules": {n: r.as_dict() for n, r in self.rules.items()},
            "scenario_history": self.scenario_history[-200:],  # keep last 200
            "policy_overrides": self._policy_overrides,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- profile tracking -----

    def record_profile_result(
        self,
        profile: str,
        won: bool,
        score: float,
    ) -> None:
        if profile not in self.profiles:
            self.profiles[profile] = ProfileStats()
        self.profiles[profile].record(won, score)

    def get_profile_ranking(self) -> list[tuple[str, ProfileStats]]:
        """Return profiles ranked by win rate, then avg score."""
        items = list(self.profiles.items())
        items.sort(key=lambda x: (x[1].win_rate, x[1].avg_score), reverse=True)
        return items

    # ----- rule effectiveness tracking -----

    def record_rule_result(
        self,
        rule_pattern: str,
        score_delta: float,
    ) -> None:
        if rule_pattern not in self.rules:
            self.rules[rule_pattern] = RuleStats()
        self.rules[rule_pattern].record(score_delta)

    def record_rules_batch(
        self,
        applied_rules: list[str],
        score_before: float,
        score_after: float,
    ) -> None:
        """Record effectiveness for a batch of rules applied together.

        Distributes the total delta evenly across rules (approximate but
        avoids needing per-rule ablation).
        """
        if not applied_rules:
            return
        total_delta = score_after - score_before
        per_rule_delta = total_delta / len(applied_rules)
        for rule in applied_rules:
            self.record_rule_result(rule, per_rule_delta)

    def get_harmful_rules(self, threshold: float = -0.02) -> list[str]:
        """Return rule patterns that consistently hurt scores."""
        return [
            name
            for name, stats in self.rules.items()
            if stats.applied_count >= 3 and stats.effectiveness < threshold
        ]

    def get_effective_rules(self, threshold: float = 0.02) -> list[str]:
        """Return rule patterns that consistently help scores."""
        return [
            name
            for name, stats in self.rules.items()
            if stats.applied_count >= 3 and stats.effectiveness > threshold
        ]

    # ----- scenario history -----

    def record_scenario(
        self,
        scenario: str,
        task: str,
        final_score: float,
        passed: bool,
        winning_profile: str,
        total_rounds: int,
    ) -> None:
        self.scenario_history.append({
            "scenario": scenario,
            "task": task[:100],
            "final_score": round(final_score, 4),
            "passed": passed,
            "winning_profile": winning_profile,
            "total_rounds": total_rounds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # ----- policy adaptation -----

    def get_policy(self, scenario: str) -> dict[str, Any]:
        """Return policy adjustments based on accumulated experience."""
        policy: dict[str, Any] = {}

        # Profile preference: suggest the historically best profile
        ranking = self.get_profile_ranking()
        if ranking:
            policy["preferred_profile"] = ranking[0][0]
            policy["profile_ranking"] = [
                {"profile": n, "win_rate": s.win_rate, "avg_score": s.avg_score}
                for n, s in ranking[:5]
            ]

        # Rule adjustments
        harmful = self.get_harmful_rules()
        if harmful:
            policy["skip_rules"] = harmful

        # Scenario-specific pass rate
        scenario_runs = [h for h in self.scenario_history if h["scenario"] == scenario]
        if len(scenario_runs) >= 3:
            pass_rate = sum(1 for h in scenario_runs if h["passed"]) / len(scenario_runs)
            policy["scenario_pass_rate"] = round(pass_rate, 4)
            if pass_rate < 0.5:
                policy["increase_max_rounds"] = True

        policy.update(self._policy_overrides)
        return policy

    def set_policy_override(self, key: str, value: Any) -> None:
        self._policy_overrides[key] = value

    # ----- summary -----

    def summary(self) -> dict[str, Any]:
        return {
            "total_profiles": len(self.profiles),
            "total_rules_tracked": len(self.rules),
            "total_scenarios": len(self.scenario_history),
            "profile_ranking": [
                {"profile": n, **s.as_dict()} for n, s in self.get_profile_ranking()
            ],
            "harmful_rules": self.get_harmful_rules(),
            "effective_rules": self.get_effective_rules(),
        }
