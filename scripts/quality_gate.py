"""Quality gate and iteration loop controller.

Manages the iterative refinement process:
  1. Score the current best candidate
  2. Determine if it passes quality thresholds
  3. If not, plan a repair round with targeted failure tags
  4. Repeat until pass or max rounds reached
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from candidates.generator import CandidateGenerator
from candidates.pool import CandidatePool, CandidateRecord
from scoring.scorer import Scorer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    """Tuneable thresholds for the quality gate."""

    pass_score: float = 0.72          # minimum final_score to pass
    hard_fail_allowed: bool = False   # if True, hard-fail candidates can still pass
    max_rounds: int = 4               # maximum iteration rounds
    improvement_floor: float = 0.005  # minimum score delta to keep iterating
    early_stop_score: float = 0.92    # skip remaining rounds if score >= this


@dataclass
class RoundLog:
    """Record of one iteration round."""

    round_number: int
    pool_size: int
    best_profile: str
    best_source_kind: str
    best_score: float
    hard_fail: bool
    failure_tags: list[str] = field(default_factory=list)
    applied_rules: list[str] = field(default_factory=list)
    all_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IterationResult:
    """Final result of the full iteration loop."""

    passed: bool
    final_text: str
    final_score: float
    final_hard_fail: bool
    total_rounds: int
    round_logs: list[RoundLog] = field(default_factory=list)
    best_record: CandidateRecord | None = None
    failure_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------

class QualityGate:
    """Decides whether a candidate passes quality thresholds."""

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()

    def check(self, record: CandidateRecord) -> tuple[bool, list[str]]:
        """Return (passed, reasons) for a single candidate."""
        reasons: list[str] = []
        s = record.score or {}
        final = float(s.get("final_score") or 0)
        hard_fail = bool(s.get("hard_fail"))

        if hard_fail and not self.config.hard_fail_allowed:
            reasons.append(f"hard_fail=True")

        if final < self.config.pass_score:
            reasons.append(f"score {final:.4f} < threshold {self.config.pass_score}")

        passed = len(reasons) == 0
        return passed, reasons

    def should_continue(
        self,
        round_number: int,
        current_score: float,
        prev_score: float,
    ) -> tuple[bool, str]:
        """Decide whether another round is warranted."""
        if round_number >= self.config.max_rounds:
            return False, f"reached max_rounds ({self.config.max_rounds})"
        if current_score >= self.config.early_stop_score:
            return False, f"score {current_score:.4f} >= early_stop {self.config.early_stop_score}"
        if round_number > 1 and (current_score - prev_score) < self.config.improvement_floor:
            return False, (
                f"improvement {current_score - prev_score:.4f} < floor {self.config.improvement_floor}"
            )
        return True, "continuing"


# ---------------------------------------------------------------------------
# Iteration Loop
# ---------------------------------------------------------------------------

class IterationLoop:
    """Runs the generate-score-repair loop up to max_rounds."""

    def __init__(
        self,
        generator: CandidateGenerator,
        gate: QualityGate | None = None,
    ):
        self.generator = generator
        self.gate = gate or QualityGate()

    def run(
        self,
        *,
        spec: dict[str, Any],
        task: str,
        source_text: str,
        scenario: str,
    ) -> IterationResult:
        """Execute the full iteration loop."""
        round_logs: list[RoundLog] = []
        best_text = source_text
        best_score = 0.0
        best_record: CandidateRecord | None = None
        failure_tags: list[str] = []

        for round_num in range(1, self.gate.config.max_rounds + 1):
            revision_mode = "rewrite" if round_num == 1 else "repair"

            pool = self.generator.generate_round(
                spec=spec,
                task=task,
                source_text=source_text,
                current_best_text=best_text,
                current_best_score=best_score,
                revision_mode=revision_mode,
                failure_tags=failure_tags,
                scenario=scenario,
            )

            if pool.count == 0:
                round_logs.append(RoundLog(
                    round_number=round_num,
                    pool_size=0,
                    best_profile="none",
                    best_source_kind="none",
                    best_score=best_score,
                    hard_fail=True,
                    failure_tags=["no_candidates"],
                ))
                break

            try:
                round_best = pool.pick_best()
            except RuntimeError:
                round_logs.append(RoundLog(
                    round_number=round_num,
                    pool_size=pool.count,
                    best_profile="none",
                    best_source_kind="none",
                    best_score=best_score,
                    hard_fail=True,
                    failure_tags=["no_scorable_candidate"],
                    all_candidates=pool.all_dicts(),
                ))
                break

            round_score = float((round_best.score or {}).get("final_score") or 0)
            round_hard_fail = bool((round_best.score or {}).get("hard_fail"))

            log = RoundLog(
                round_number=round_num,
                pool_size=pool.count,
                best_profile=round_best.profile,
                best_source_kind=round_best.source_kind,
                best_score=round_score,
                hard_fail=round_hard_fail,
                failure_tags=list(round_best.failure_tags),
                applied_rules=list(round_best.applied_rules),
                all_candidates=pool.all_dicts(),
            )
            round_logs.append(log)

            # Update best-so-far (only if better)
            if round_score > best_score:
                best_score = round_score
                best_text = round_best.text
                best_record = round_best
                failure_tags = list(round_best.failure_tags)

            # Check quality gate
            passed, _reasons = self.gate.check(round_best)
            if passed:
                return IterationResult(
                    passed=True,
                    final_text=best_text,
                    final_score=best_score,
                    final_hard_fail=False,
                    total_rounds=round_num,
                    round_logs=round_logs,
                    best_record=best_record,
                    failure_tags=failure_tags,
                )

            # Check if we should keep iterating
            prev_score = round_logs[-2].best_score if len(round_logs) >= 2 else 0.0
            should_go, _reason = self.gate.should_continue(round_num, best_score, prev_score)
            if not should_go:
                break

        return IterationResult(
            passed=best_score >= self.gate.config.pass_score,
            final_text=best_text,
            final_score=best_score,
            final_hard_fail=best_record is not None and bool((best_record.score or {}).get("hard_fail")),
            total_rounds=len(round_logs),
            round_logs=round_logs,
            best_record=best_record,
            failure_tags=failure_tags,
        )
