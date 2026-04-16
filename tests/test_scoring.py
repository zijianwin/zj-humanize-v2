"""Tests for scoring system."""

from __future__ import annotations

import pytest

from scoring.scorer import (
    Scorer,
    CandidateScore,
    ScenarioWeights,
    SCENARIO_WEIGHTS,
    _compact_char_count,
    _normalize_for_similarity,
    _weighted_average,
)
from heuristics.detector import TemplatePhraseDetector


class TestScenarioDetection:
    def setup_method(self):
        self.scorer = Scorer()

    def test_email_scenario(self):
        assert self.scorer.detect_scenario({"task": "回复客户邮件"}) == "email"
        assert self.scorer.detect_scenario({"task": "send email to HR"}) == "email"

    def test_wechat_scenario(self):
        assert self.scorer.detect_scenario({"task": "微信回复客户"}) == "wechat"
        assert self.scorer.detect_scenario({"task": "飞书消息"}) == "wechat"

    def test_longform_scenario(self):
        assert self.scorer.detect_scenario({"task": "小红书种草文案"}) == "longform"
        assert self.scorer.detect_scenario({"task": "自媒体文章"}) == "longform"

    def test_service_scenario(self):
        assert self.scorer.detect_scenario({"task": "售后客服回复"}) == "service"
        assert self.scorer.detect_scenario({"task": "处理退款投诉"}) == "service"

    def test_default_scenario(self):
        assert self.scorer.detect_scenario({"task": "写一段话"}) == "default"
        assert self.scorer.detect_scenario({}) == "default"


class TestScenarioWeights:
    def test_all_scenarios_exist(self):
        for name in ("email", "wechat", "longform", "service", "default"):
            assert name in SCENARIO_WEIGHTS

    def test_weights_as_list(self):
        w = ScenarioWeights()
        items = w.as_list()
        assert len(items) == 20  # 15 general + 5 longform-specific
        assert all(isinstance(n, str) and isinstance(v, float) for n, v in items)


class TestScoring:
    def setup_method(self):
        self.scorer = Scorer()

    def test_basic_scoring(self):
        spec = {"task": "微信回复"}
        candidate = "好的，退款正在处理中，预计3个工作日内到账。有问题随时找我。"
        source = "尊敬的客户您好，关于您的退款申请，我们正在积极处理中。"
        result = self.scorer.score(spec, candidate, source)
        assert isinstance(result, CandidateScore)
        assert 0 <= result.final_score <= 1
        assert isinstance(result.notes, list)
        assert isinstance(result.rule_breakdown, dict)

    def test_empty_candidate_hard_fail(self):
        spec = {"task": "测试"}
        result = self.scorer.score(spec, "", "原始文本")
        assert result.hard_fail

    def test_must_include_check(self):
        spec = {
            "task": "测试",
            "hard_constraints": {"must_include": ["退款", "3个工作日"]},
        }
        candidate_pass = "退款会在3个工作日内处理"
        candidate_fail = "已经在处理了"
        r1 = self.scorer.score(spec, candidate_pass, "")
        r2 = self.scorer.score(spec, candidate_fail, "")
        assert not r1.hard_fail or r1.final_score > r2.final_score
        assert r2.hard_fail

    def test_banned_phrases(self):
        spec = {
            "task": "测试",
            "hard_constraints": {"banned_phrases": ["竭诚为您服务"]},
        }
        candidate_clean = "有问题随时找我"
        candidate_bad = "我们将竭诚为您服务"
        r1 = self.scorer.score(spec, candidate_clean, "")
        r2 = self.scorer.score(spec, candidate_bad, "")
        assert r1.final_score >= r2.final_score

    def test_length_constraints(self):
        spec = {
            "task": "测试",
            "hard_constraints": {"min_chars": 20, "max_chars": 50},
        }
        too_short = "短"
        just_right = "这是一段长度适中的测试文本，刚好在范围内。"
        r_short = self.scorer.score(spec, too_short, "")
        r_ok = self.scorer.score(spec, just_right, "")
        assert r_ok.final_score > r_short.final_score

    def test_scenario_affects_weights(self):
        spec_email = {"task": "回复邮件"}
        spec_wechat = {"task": "微信回复"}
        w_email = self.scorer.get_weights("email")
        w_wechat = self.scorer.get_weights("wechat")
        assert w_email.email_shape > w_wechat.email_shape

    def test_template_details_populated(self):
        detector = TemplatePhraseDetector()
        detector.add_phrase("竭诚为您服务")
        scorer = Scorer(detector=detector)
        spec = {"task": "测试"}
        result = scorer.score(spec, "我们将竭诚为您服务！", "")
        assert isinstance(result.template_details, list)


class TestHelpers:
    def test_compact_char_count(self):
        assert _compact_char_count("你好 世界") == 4
        assert _compact_char_count("  空格  ") == 2
        assert _compact_char_count("") == 0

    def test_normalize_for_similarity(self):
        n = _normalize_for_similarity("你好，世界！")
        assert "，" not in n
        assert "！" not in n
        assert "你好" in n

    def test_weighted_average(self):
        parts = [("a", 1.0, 0.5), ("b", 0.5, 0.5)]
        val, breakdown = _weighted_average(parts)
        assert abs(val - 0.75) < 0.001
        assert breakdown["a"] == 1.0
        assert breakdown["b"] == 0.5

    def test_weighted_average_zero_weights(self):
        parts = [("a", 1.0, 0.0), ("b", 0.5, 0.0)]
        val, _ = _weighted_average(parts)
        assert val == 0.0
