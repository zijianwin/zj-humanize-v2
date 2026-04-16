"""Output rendering: converts iteration results into human-readable reports."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scripts.quality_gate import IterationResult, RoundLog


class OutputRenderer:
    """Renders iteration results into structured output."""

    def __init__(self, *, verbose: bool = False):
        self.verbose = verbose

    # ----- Main render methods -----

    def render_text(self, result: IterationResult, spec: dict[str, Any]) -> str:
        """Render a human-readable text report."""
        scenario = str(spec.get("scenario", ""))
        if scenario == "longform":
            return self._render_longform_diagnostic(result, spec)
        return self._render_standard_text(result, spec)

    def _render_standard_text(self, result: IterationResult, spec: dict[str, Any]) -> str:
        """Standard text report for short-form content."""
        lines: list[str] = []
        task = str(spec.get("task", ""))

        lines.append("=" * 60)
        lines.append("  Humanize v2 — 去AI味处理结果")
        lines.append("=" * 60)
        lines.append("")

        if task:
            lines.append(f"📋 任务：{task}")
        lines.append(f"📊 最终得分：{result.final_score:.4f}")
        lines.append(f"✅ 通过质量门：{'是' if result.passed else '否'}")
        lines.append(f"🔄 迭代轮次：{result.total_rounds}")

        if result.best_record:
            lines.append(f"🏆 胜出方案：{result.best_record.profile} ({result.best_record.source_kind})")

        if result.final_hard_fail:
            lines.append("⚠️  存在硬性失败项")

        lines.append("")
        lines.append("─" * 60)
        lines.append("📝 输出文本：")
        lines.append("─" * 60)
        lines.append(result.final_text)
        lines.append("")

        if result.failure_tags:
            lines.append("─" * 60)
            lines.append("⚠️  剩余问题标签：")
            for tag in result.failure_tags:
                lines.append(f"  · {tag}")
            lines.append("")

        if self.verbose and result.round_logs:
            lines.append("─" * 60)
            lines.append("📈 迭代详情：")
            lines.append("─" * 60)
            for log in result.round_logs:
                lines.append(self._render_round(log))
            lines.append("")

        if self.verbose and result.best_record and result.best_record.score:
            lines.append("─" * 60)
            lines.append("📊 评分细节：")
            lines.append("─" * 60)
            breakdown = result.best_record.score.get("rule_breakdown", {})
            for dim, val in sorted(breakdown.items(), key=lambda x: x[1]):
                bar = "█" * int(val * 20)
                lines.append(f"  {dim:30s} {val:.4f}  {bar}")
            notes = result.best_record.score.get("notes", [])
            if notes:
                lines.append("")
                lines.append("  备注：")
                for note in notes:
                    lines.append(f"    · {note}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def render_json(self, result: IterationResult, spec: dict[str, Any]) -> str:
        """Render a structured JSON report."""
        data = self._to_dict(result, spec)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def render_brief(self, result: IterationResult) -> str:
        """Render a minimal one-line summary."""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        profile = result.best_record.profile if result.best_record else "none"
        return (
            f"{status} | score={result.final_score:.4f} | "
            f"rounds={result.total_rounds} | profile={profile}"
        )

    # ----- Export -----

    def save_report(
        self,
        result: IterationResult,
        spec: dict[str, Any],
        output_dir: Path,
        *,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """Save reports to files. Returns list of output paths."""
        formats = formats or ["text", "json"]
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved: list[Path] = []

        if "text" in formats:
            path = output_dir / f"report_{timestamp}.txt"
            path.write_text(self.render_text(result, spec), encoding="utf-8")
            saved.append(path)

        if "json" in formats:
            path = output_dir / f"report_{timestamp}.json"
            path.write_text(self.render_json(result, spec), encoding="utf-8")
            saved.append(path)

        return saved

    # ----- Longform diagnostic report -----

    def _render_longform_diagnostic(self, result: IterationResult, spec: dict[str, Any]) -> str:
        """Render a diagnostic report for longform articles."""
        lines: list[str] = []
        task = str(spec.get("task", ""))

        lines.append("=" * 60)
        lines.append("  Humanize v2 — 长文AI味诊断报告")
        lines.append("=" * 60)
        lines.append("")

        if task:
            lines.append(f"📋 任务：{task}")
        lines.append(f"📊 综合评分：{result.final_score:.4f}")

        # Determine pass level with emoji
        if result.final_score >= 0.85:
            level = "✅ 优秀 — AI味极低"
        elif result.final_score >= 0.72:
            level = "🟢 良好 — AI味可接受"
        elif result.final_score >= 0.55:
            level = "🟡 一般 — 有明显AI味，建议修改"
        else:
            level = "🔴 较差 — AI味严重，需要重写"
        lines.append(f"📈 评级：{level}")
        lines.append("")

        # Extract scoring details
        if result.best_record and result.best_record.score:
            breakdown = result.best_record.score.get("rule_breakdown", {})
            notes = result.best_record.score.get("notes", [])

            # Group dimensions
            longform_dims = [
                ("negation_flip_density", "「不是A而是B」句式密度"),
                ("word_repetition", "近距离用词重复"),
                ("parallel_structure_density", "排比结构密度"),
                ("summary_sentence_pattern", "AI式总结金句"),
                ("explanatory_redundancy", "同义解释冗余"),
            ]
            general_dims = [
                ("template_tone", "模板短语/AI套话"),
                ("source_template_reduction", "原文模板消除"),
                ("anti_repetition", "N-gram重复片段"),
                ("sentence_splice", "句式拼接问题"),
                ("formatting", "格式规范性"),
            ]

            # Render longform-specific scores
            lines.append("─" * 60)
            lines.append("🔍 长文专项检测：")
            lines.append("─" * 60)
            problem_count = 0
            for dim_key, dim_label in longform_dims:
                val = breakdown.get(dim_key, 1.0)
                if val >= 0.9:
                    icon = "✅"
                elif val >= 0.7:
                    icon = "🟡"
                else:
                    icon = "🔴"
                    problem_count += 1
                bar = "█" * int(val * 15) + "░" * (15 - int(val * 15))
                lines.append(f"  {icon} {dim_label:20s}  {bar}  {val:.2f}")

            # Render general scores
            lines.append("")
            lines.append("─" * 60)
            lines.append("📋 通用检测：")
            lines.append("─" * 60)
            for dim_key, dim_label in general_dims:
                val = breakdown.get(dim_key, 1.0)
                if val >= 0.9:
                    icon = "✅"
                elif val >= 0.7:
                    icon = "🟡"
                else:
                    icon = "🔴"
                bar = "█" * int(val * 15) + "░" * (15 - int(val * 15))
                lines.append(f"  {icon} {dim_label:20s}  {bar}  {val:.2f}")

            # Render specific notes as actionable advice
            if notes:
                lines.append("")
                lines.append("─" * 60)
                lines.append("💡 具体问题与修改建议：")
                lines.append("─" * 60)
                for i, note in enumerate(notes, 1):
                    lines.append(f"  {i}. {note}")
                lines.append("")

        # Verbose: show all dimension scores
        if self.verbose and result.best_record and result.best_record.score:
            breakdown = result.best_record.score.get("rule_breakdown", {})
            lines.append("─" * 60)
            lines.append("📊 全维度评分明细：")
            lines.append("─" * 60)
            for dim, val in sorted(breakdown.items(), key=lambda x: x[1]):
                bar = "█" * int(val * 20)
                lines.append(f"  {dim:35s} {val:.4f}  {bar}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ----- Internal helpers -----

    def _render_round(self, log: RoundLog) -> str:
        status = "✅" if not log.hard_fail else "⚠️"
        lines = [
            f"  Round {log.round_number}: {status} "
            f"score={log.best_score:.4f} | "
            f"pool={log.pool_size} | "
            f"best={log.best_profile}"
        ]
        if log.failure_tags:
            lines.append(f"    tags: {', '.join(log.failure_tags)}")
        if log.applied_rules:
            lines.append(f"    rules: {', '.join(log.applied_rules[:5])}")
        return "\n".join(lines)

    def _to_dict(self, result: IterationResult, spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": str(spec.get("task", "")),
            "scenario": str(spec.get("scenario", "")),
            "passed": result.passed,
            "final_text": result.final_text,
            "final_score": round(result.final_score, 6),
            "final_hard_fail": result.final_hard_fail,
            "total_rounds": result.total_rounds,
            "failure_tags": result.failure_tags,
            "best_record": result.best_record.as_dict() if result.best_record else None,
            "round_logs": [
                {
                    "round": l.round_number,
                    "pool_size": l.pool_size,
                    "best_profile": l.best_profile,
                    "best_source_kind": l.best_source_kind,
                    "best_score": round(l.best_score, 6),
                    "hard_fail": l.hard_fail,
                    "failure_tags": l.failure_tags,
                    "applied_rules": l.applied_rules,
                    "all_candidates": l.all_candidates if self.verbose else [],
                }
                for l in result.round_logs
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
