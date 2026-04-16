"""Humanize v2 — 去AI味全链路处理引擎。

Main orchestrator that wires together all modules:
  heuristics → scoring → candidates → iteration loop → output

Usage (CLI):
    python humanize.py brief.json
    python humanize.py brief.json --verbose --output-dir ./output

Usage (as library):
    from humanize import HumanizeEngine
    engine = HumanizeEngine()
    result = engine.run(spec, source_text)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from heuristics.engine import ReplacementEngine
from heuristics.detector import TemplatePhraseDetector
from scoring.scorer import Scorer
from candidates.generator import CandidateGenerator
from scripts.quality_gate import QualityGate, GateConfig, IterationLoop, IterationResult
from scripts.strategy_state import StrategyState
from scripts.style_learner import StyleLearner
from feedback.collector import FeedbackStore, FeedbackRecord
from reporting.renderer import OutputRenderer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_RULES_DIR = _THIS_DIR / "heuristics" / "rules"
_PHRASES_PATH = _RULES_DIR / "template_phrases.yaml"
_STATE_DIR = _THIS_DIR / ".state"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HumanizeEngine:
    """Top-level engine that coordinates all subsystems."""

    def __init__(
        self,
        *,
        rules_dir: Path | None = None,
        phrases_path: Path | None = None,
        state_dir: Path | None = None,
        model_bundle: dict[str, Any] | None = None,
        llm_caller: Any | None = None,
        gate_config: GateConfig | None = None,
        verbose: bool = False,
    ):
        rules_dir = rules_dir or _RULES_DIR
        phrases_path = phrases_path or _PHRASES_PATH
        state_dir = state_dir or _STATE_DIR
        state_dir.mkdir(parents=True, exist_ok=True)

        # Subsystems
        self.replacement_engine = ReplacementEngine(rules_dir)
        self.detector = TemplatePhraseDetector(phrases_path)
        self.scorer = Scorer(detector=self.detector, model_bundle=model_bundle)
        self.generator = CandidateGenerator(
            scorer=self.scorer,
            engine=self.replacement_engine,
            llm_caller=llm_caller,
        )
        self.gate = QualityGate(gate_config or GateConfig())
        self.loop = IterationLoop(self.generator, self.gate)
        self.strategy = StrategyState(state_dir / "strategy.json")
        self.style_learner = StyleLearner(state_dir / "style_profile.json")
        self.feedback_store = FeedbackStore(state_dir / "feedback.json")
        self.renderer = OutputRenderer(verbose=verbose)

    def run(
        self,
        spec: dict[str, Any],
        source_text: str,
        *,
        scenario: str | None = None,
    ) -> IterationResult:
        """Run the full humanize pipeline and return the iteration result."""
        task = str(spec.get("task", ""))

        # Auto-detect scenario
        if scenario is None:
            scenario = self.scorer.detect_scenario(spec)
        spec["scenario"] = scenario

        # Get policy from strategy state
        policy = self.strategy.get_policy(scenario)
        if policy.get("increase_max_rounds") and self.gate.config.max_rounds < 6:
            self.gate.config.max_rounds += 1

        # Inject style hints if available
        style_hints = self.style_learner.get_style_hints()
        if style_hints:
            existing_notes = spec.get("style_notes") or []
            if style_hints.get("tone"):
                existing_notes.append(f"tone: {style_hints['tone']}")
            spec["style_notes"] = existing_notes

        # Run the iteration loop
        result = self.loop.run(
            spec=spec,
            task=task,
            source_text=source_text,
            scenario=scenario,
        )

        # Record strategy data
        if result.best_record:
            # Record profile result
            winning_profile = result.best_record.profile
            for log in result.round_logs:
                for cand in log.all_candidates:
                    profile = cand.get("profile", "")
                    score = float((cand.get("score") or {}).get("final_score") or 0)
                    won = profile == winning_profile
                    self.strategy.record_profile_result(profile, won, score)

            # Record rule effectiveness
            if result.best_record.applied_rules:
                initial_score_obj = self.scorer.score(spec, source_text, source_text, scenario=scenario)
                baseline = initial_score_obj.final_score
                self.strategy.record_rules_batch(
                    result.best_record.applied_rules,
                    score_before=baseline,
                    score_after=result.final_score,
                )

            self.strategy.record_scenario(
                scenario=scenario,
                task=task,
                final_score=result.final_score,
                passed=result.passed,
                winning_profile=winning_profile,
                total_rounds=result.total_rounds,
            )
            self.strategy.save()

        return result

    def run_and_render(
        self,
        spec: dict[str, Any],
        source_text: str,
        *,
        scenario: str | None = None,
        output_format: str = "text",
    ) -> str:
        """Run pipeline and return rendered output."""
        result = self.run(spec, source_text, scenario=scenario)
        if output_format == "json":
            return self.renderer.render_json(result, spec)
        elif output_format == "brief":
            return self.renderer.render_brief(result)
        return self.renderer.render_text(result, spec)

    def collect_feedback(
        self,
        result: IterationResult,
        spec: dict[str, Any],
        *,
        rating: str = "",
        comment: str = "",
        manual_edit: str = "",
    ) -> None:
        """Collect user feedback for a run."""
        source_text = str(spec.get("source_text", ""))
        record = FeedbackRecord(
            task=str(spec.get("task", "")),
            scenario=str(spec.get("scenario", "")),
            source_text=source_text,
            output_text=result.final_text,
            final_score=result.final_score,
            rating=rating,
            comment=comment,
            manual_edit=manual_edit,
            failure_areas=result.failure_tags,
            winning_profile=result.best_record.profile if result.best_record else "",
        )
        self.feedback_store.add(record)

        # If user provided a manual edit, learn from it
        if manual_edit.strip():
            self.style_learner.learn_from_edit_pair(result.final_text, manual_edit)
            self.style_learner.save()

    def get_status(self) -> dict[str, Any]:
        """Return system status summary."""
        return {
            "replacement_rules": self.replacement_engine.rule_count,
            "template_phrases": self.detector.phrase_count,
            "strategy_summary": self.strategy.summary(),
            "feedback_count": self.feedback_store.total_count,
            "style_profile": self.style_learner.profile.as_dict(),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_brief(path: str) -> tuple[dict[str, Any], str]:
    """Load a brief.json file and extract spec + source_text."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        source = str(data.get("source_text", "") or data.get("text", "") or data.get("content", ""))
        spec = {k: v for k, v in data.items() if k not in ("source_text", "text", "content")}
        return spec, source
    raise ValueError(f"Invalid brief format: expected a JSON object, got {type(data).__name__}")


def _parse_text_request(text: str) -> tuple[dict[str, Any], str]:
    """Parse a free-form --text request into spec + source_text.

    Supports natural language like:
      "帮我改写这段客服回复，必须保留'退款'和'3个工作日'，不超过120字。原文：尊敬的客户您好..."
    """
    spec: dict[str, Any] = {}
    source_text = ""
    hard_constraints: dict[str, Any] = {}

    # Extract source text from common delimiters
    for marker in ("原文：", "原文:", "原稿：", "原稿:", "正文：", "正文:",
                    "原始文本：", "原始文本:", "draft:", "draft：",
                    "原文如下：", "原文如下:", "内容：", "内容:"):
        if marker in text:
            idx = text.index(marker) + len(marker)
            source_text = text[idx:].strip()
            text = text[:idx - len(marker)].strip()
            break

    if not source_text:
        # If no marker found, treat the whole text as source if it's long enough,
        # or as a task description if short
        if len(text) > 80:
            source_text = text
        else:
            spec["task"] = text
            return spec, source_text

    # Extract must_include
    mi_patterns = [
        r"必须保留[：:]?\s*['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
        r"保留[：:]?\s*['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
    ]
    must_include: list[str] = []
    for pat in mi_patterns:
        must_include.extend(re.findall(pat, text))
    if must_include:
        hard_constraints["must_include"] = must_include

    # Extract banned phrases
    ban_patterns = [
        r"不要[用使]?[：:]?\s*['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
        r"禁[用止][：:]?\s*['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
        r"避免[：:]?\s*['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]+)['\"\u2018\u2019\u201c\u201d]",
    ]
    banned: list[str] = []
    for pat in ban_patterns:
        banned.extend(re.findall(pat, text))
    if banned:
        hard_constraints["banned_phrases"] = banned

    # Extract max_chars
    max_match = re.search(r"不超过(\d+)字|最多(\d+)字|(\d+)字以内|max[_\s]*(\d+)", text)
    if max_match:
        val = next(g for g in max_match.groups() if g is not None)
        hard_constraints["max_chars"] = int(val)

    # Extract min_chars
    min_match = re.search(r"至少(\d+)字|不少于(\d+)字|(\d+)字以上|min[_\s]*(\d+)", text)
    if min_match:
        val = next(g for g in min_match.groups() if g is not None)
        hard_constraints["min_chars"] = int(val)

    if hard_constraints:
        spec["hard_constraints"] = hard_constraints

    # The remaining text before the marker becomes the task
    task_text = text.strip()
    if task_text:
        spec["task"] = task_text

    return spec, source_text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Humanize v2 — 去AI味全链路处理引擎",
    )
    parser.add_argument("brief", nargs="?", default=None, help="Path to brief.json file")
    parser.add_argument("--text", "-t", type=str, default=None, help="Direct text input (natural language request)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed scoring breakdown")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Save report files to this directory")
    parser.add_argument("--format", choices=["text", "json", "brief"], default="text", help="Output format")
    parser.add_argument("--scenario", type=str, default=None, help="Force a specific scenario")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override max iteration rounds")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")

    args = parser.parse_args(argv)

    gate_config = GateConfig()
    if args.max_rounds:
        gate_config.max_rounds = args.max_rounds

    engine = HumanizeEngine(
        gate_config=gate_config,
        verbose=args.verbose,
    )

    if args.status:
        status = engine.get_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    # Determine input source
    if args.text:
        spec, source_text = _parse_text_request(args.text)
    elif args.brief:
        spec, source_text = _load_brief(args.brief)
    else:
        print("Error: provide either --text or a brief.json path", file=sys.stderr)
        sys.exit(1)

    if not source_text:
        print("Error: no source_text found in input", file=sys.stderr)
        sys.exit(1)

    result = engine.run(spec, source_text, scenario=args.scenario)
    output = engine.renderer.render_text(result, spec) if args.format == "text" else (
        engine.renderer.render_json(result, spec) if args.format == "json" else
        engine.renderer.render_brief(result)
    )
    # Use utf-8 to avoid encoding errors on Windows terminals (GBK)
    sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()

    if args.output_dir:
        paths = engine.renderer.save_report(
            result, spec, Path(args.output_dir), formats=["text", "json"]
        )
        for p in paths:
            print(f"Saved: {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
