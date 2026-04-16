"""Candidate generator: model + heuristic collaboration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from heuristics.engine import ReplacementEngine
from scoring.scorer import Scorer
from candidates.pool import CandidatePool, CandidateRecord


class CandidateGenerator:
    """Generates candidates via model calls and heuristic rewrites, always collaborating."""

    def __init__(
        self,
        scorer: Scorer,
        engine: ReplacementEngine,
        llm_caller: Any | None = None,  # optional LLM call function
    ):
        self.scorer = scorer
        self.engine = engine
        self.llm_caller = llm_caller

    def generate_round(
        self,
        *,
        spec: dict[str, Any],
        task: str,
        source_text: str,
        current_best_text: str,
        current_best_score: float,
        revision_mode: str,
        failure_tags: list[str],
        scenario: str,
    ) -> CandidatePool:
        """Generate a pool of candidates for one round.

        Model and heuristic candidates always co-exist when possible.
        For longform scenario, always produce a diagnostic candidate even
        when heuristic rules don't change the text.
        """
        pool = CandidatePool()
        hard_constraints = spec.get("hard_constraints") or {}

        # 1. Heuristic candidates (always generated)
        base_text = current_best_text if revision_mode == "repair" else source_text
        heuristic_produced = False
        for style in ("natural", "balanced"):
            context = task
            result = self.engine.apply(base_text, style=style, context=context)
            if result.text.strip() != current_best_text.strip():
                score_obj = self.scorer.score(spec, result.text, source_text, scenario=scenario)
                tags = _extract_tags(result.text, score_obj.as_dict(), current_best_score)
                pool.add(
                    profile=f"heuristic-{style}",
                    source_kind="heuristic",
                    text=result.text,
                    score=score_obj.as_dict(),
                    failure_tags=tags,
                    applied_rules=result.applied_rules,
                )
                heuristic_produced = True

        # 2. For longform scenario: if no heuristic changes, generate a
        #    diagnostic-only candidate using the original text scored with
        #    longform-specific dimensions.
        if not heuristic_produced and scenario == "longform":
            score_obj = self.scorer.score(spec, source_text, source_text, scenario=scenario)
            tags = _extract_tags(source_text, score_obj.as_dict(), current_best_score)
            pool.add(
                profile="longform-diagnostic",
                source_kind="diagnostic",
                text=source_text,
                score=score_obj.as_dict(),
                failure_tags=tags,
                applied_rules=[],
            )

        # 3. Model candidates (when LLM is available)
        if self.llm_caller is not None:
            try:
                model_text = self._call_model(
                    task=task,
                    hard_constraints=hard_constraints,
                    source_text=source_text,
                    current_best_text=current_best_text,
                    revision_mode=revision_mode,
                    failure_tags=failure_tags,
                )
                if model_text and model_text.strip() != current_best_text.strip():
                    profile = "direct-repair" if revision_mode == "repair" else "direct-rewrite"
                    score_obj = self.scorer.score(spec, model_text, source_text, scenario=scenario)
                    tags = _extract_tags(model_text, score_obj.as_dict(), current_best_score)
                    pool.add(
                        profile=profile,
                        source_kind="model_repair" if revision_mode == "repair" else "model_direct",
                        text=model_text,
                        score=score_obj.as_dict(),
                        failure_tags=tags,
                    )
            except Exception as exc:
                pool.add(
                    profile="direct-rewrite",
                    source_kind="model_direct",
                    text="",
                    score={"final_score": 0.0, "hard_fail": True, "notes": [f"generation error: {exc}"]},
                    failure_tags=["generation_error"],
                    error=str(exc),
                )

        return pool

    def _call_model(
        self,
        *,
        task: str,
        hard_constraints: dict[str, Any],
        source_text: str,
        current_best_text: str,
        revision_mode: str,
        failure_tags: list[str],
    ) -> str:
        """Call the LLM to generate a candidate. Delegates to self.llm_caller."""
        if self.llm_caller is None:
            return ""
        return self.llm_caller(
            task=task,
            hard_constraints=hard_constraints,
            source_text=source_text,
            current_best_text=current_best_text,
            revision_mode=revision_mode,
            failure_tags=failure_tags,
        )


def _extract_tags(text: str, score: dict[str, Any], baseline_score: float) -> list[str]:
    tags: list[str] = []
    notes = [str(n).lower() for n in score.get("notes") or []]
    if any("missing must_include" in n for n in notes):
        tags.append("missing_must_include")
    if any("contains template phrases" in n for n in notes):
        tags.append("template_tone")
    if any("retains source template phrases" in n for n in notes):
        tags.append("source_template_carryover")
    if any("rewrite too similar" in n or "rewrite still very close" in n for n in notes):
        tags.append("too_similar")
    if any("sentence splice issue" in n for n in notes):
        tags.append("bad_splice")
    if any("placeholder" in n for n in notes):
        tags.append("placeholder_output")
    if any("over-compressed" in n or "drops too much" in n for n in notes):
        tags.append("overcompressed")
    if score.get("hard_fail"):
        tags.append("hard_fail")
    if float(score.get("final_score") or 0) < baseline_score:
        tags.append("no_improvement")
    # deduplicate
    seen: set[str] = set()
    return [t for t in tags if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
