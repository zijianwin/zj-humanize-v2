"""Tests for heuristics engine and detector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from heuristics.engine import ReplacementEngine, ReplacementRule, ReplacementResult
from heuristics.detector import (
    TemplatePhraseDetector,
    TemplatePhrase,
    PhraseSeverity,
    SEVERITY_PENALTY,
)


# ---------------------------------------------------------------------------
# ReplacementRule
# ---------------------------------------------------------------------------

class TestReplacementRule:
    def test_from_dict_basic(self):
        data = {"pattern": "非常感谢", "natural": "谢谢", "balanced": "感谢"}
        rule = ReplacementRule.from_dict(data)
        assert rule.pattern == "非常感谢"
        assert rule.natural == "谢谢"
        assert rule.balanced == "感谢"
        assert rule.fuzzy is False
        assert rule.is_regex is False

    def test_from_dict_fuzzy(self):
        data = {"pattern": "竭诚为您服务", "natural": "", "fuzzy": True, "threshold": 0.8}
        rule = ReplacementRule.from_dict(data)
        assert rule.fuzzy is True
        assert rule.threshold == 0.8

    def test_from_dict_with_tags(self):
        data = {"pattern": "test", "natural": "t", "tags": ["客服腔", "模板"]}
        rule = ReplacementRule.from_dict(data)
        assert rule.tags == ["客服腔", "模板"]

    def test_from_dict_contexts_exempt(self):
        data = {"pattern": "p", "natural": "n", "contexts_exempt": ["售后", "客服"]}
        rule = ReplacementRule.from_dict(data)
        assert "售后" in rule.contexts_exempt


# ---------------------------------------------------------------------------
# ReplacementEngine
# ---------------------------------------------------------------------------

class TestReplacementEngine:
    def _make_engine_with_rules(self, rules: list[dict]) -> ReplacementEngine:
        engine = ReplacementEngine()
        for data in rules:
            engine.rules.append(ReplacementRule.from_dict(data))
        return engine

    def test_exact_replacement(self):
        engine = self._make_engine_with_rules([
            {"pattern": "非常感谢您的耐心等待", "natural": "久等了", "balanced": "感谢等待"},
        ])
        result = engine.apply("非常感谢您的耐心等待，退款正在处理。")
        assert "久等了" in result.text
        assert "非常感谢您的耐心等待" in result.applied_rules

    def test_balanced_style(self):
        engine = self._make_engine_with_rules([
            {"pattern": "非常感谢您的耐心等待", "natural": "久等了", "balanced": "感谢等待"},
        ])
        result = engine.apply("非常感谢您的耐心等待", style="balanced")
        assert "感谢等待" in result.text

    def test_context_exemption(self):
        engine = self._make_engine_with_rules([
            {"pattern": "感谢您", "natural": "谢谢", "contexts_exempt": ["邮件"]},
        ])
        result = engine.apply("感谢您的支持", context="邮件回复")
        assert "感谢您" in result.text  # should be exempt
        assert "感谢您" in result.skipped_rules

    def test_no_match_returns_original(self):
        engine = self._make_engine_with_rules([
            {"pattern": "不存在的短语", "natural": "替换"},
        ])
        original = "这是一段正常文本"
        result = engine.apply(original)
        assert result.text == original
        assert len(result.applied_rules) == 0

    def test_regex_replacement(self):
        engine = self._make_engine_with_rules([
            {"pattern": r"非常[感谢激动高兴]+", "natural": "谢谢", "is_regex": True},
        ])
        result = engine.apply("非常感谢你的帮助")
        assert "谢谢" in result.text

    def test_priority_ordering(self):
        engine = self._make_engine_with_rules([
            {"pattern": "AB", "natural": "X", "priority": 1},
            {"pattern": "ABC", "natural": "Y", "priority": 10},
        ])
        engine.rules.sort(key=lambda r: r.priority, reverse=True)
        result = engine.apply("ABC")
        assert result.text == "Y"

    def test_yaml_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """rules:
  - pattern: "测试短语"
    natural: "替换结果"
    balanced: "平衡结果"
"""
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text(yaml_content, encoding="utf-8")
            engine = ReplacementEngine(Path(tmpdir))
            assert engine.rule_count >= 1
            result = engine.apply("包含测试短语的文本")
            assert "替换结果" in result.text


# ---------------------------------------------------------------------------
# TemplatePhraseDetector
# ---------------------------------------------------------------------------

class TestTemplatePhraseDetector:
    def _make_detector(self, phrases: list[dict]) -> TemplatePhraseDetector:
        detector = TemplatePhraseDetector()
        for data in phrases:
            detector.phrases.append(TemplatePhrase.from_dict(data))
        return detector

    def test_basic_detection(self):
        detector = self._make_detector([
            {"phrase": "竭诚为您服务", "severity": "critical"},
            {"phrase": "温馨提示", "severity": "high"},
        ])
        result = detector.detect("我们将竭诚为您服务，温馨提示请注意安全。")
        assert result.hit_count == 2
        assert result.has_critical
        assert result.total_penalty > 0

    def test_no_hits(self):
        detector = self._make_detector([
            {"phrase": "不存在的短语", "severity": "high"},
        ])
        result = detector.detect("这是正常的文本。")
        assert result.hit_count == 0
        assert result.total_penalty == 0.0

    def test_context_exemption(self):
        detector = self._make_detector([
            {"phrase": "温馨提示", "severity": "high", "contexts_exempt": ["物业"]},
        ])
        result = detector.detect("温馨提示请注意安全。", context="物业通知")
        assert result.hit_count == 0
        details = [d for d in result.details if d["status"] == "exempt"]
        assert len(details) == 1

    def test_severity_penalties(self):
        detector = self._make_detector([
            {"phrase": "A", "severity": "critical"},
            {"phrase": "B", "severity": "low"},
        ])
        result = detector.detect("A and B")
        crit_pen = SEVERITY_PENALTY[PhraseSeverity.CRITICAL]
        low_pen = SEVERITY_PENALTY[PhraseSeverity.LOW]
        assert abs(result.total_penalty - (crit_pen + low_pen)) < 0.001

    def test_source_reduction_score(self):
        detector = self._make_detector([
            {"phrase": "竭诚为您服务", "severity": "critical"},
            {"phrase": "感谢您", "severity": "moderate"},
        ])
        source = "竭诚为您服务，感谢您的支持"
        candidate_clean = "谢谢你的支持"
        candidate_dirty = "竭诚为您服务，感谢您的支持"
        assert detector.source_reduction_score(source, candidate_clean) == 1.0
        assert detector.source_reduction_score(source, candidate_dirty) < 0.1

    def test_add_phrase(self):
        detector = TemplatePhraseDetector()
        detector.add_phrase("测试短语", severity=PhraseSeverity.HIGH)
        assert detector.phrase_count == 1
        result = detector.detect("这里有测试短语")
        assert result.hit_count == 1

    def test_yaml_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = """phrases:
  - phrase: "来自yaml的短语"
    severity: "moderate"
    category: "test"
"""
            yaml_path = Path(tmpdir) / "phrases.yaml"
            yaml_path.write_text(yaml_content, encoding="utf-8")
            detector = TemplatePhraseDetector(yaml_path)
            assert detector.phrase_count >= 1
            result = detector.detect("检测来自yaml的短语")
            assert result.hit_count == 1
