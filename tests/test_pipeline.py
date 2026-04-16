"""Tests for quality gate, strategy state, style learner, feedback."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from candidates.pool import CandidateRecord
from scripts.quality_gate import QualityGate, GateConfig
from scripts.strategy_state import StrategyState, ProfileStats, RuleStats
from scripts.style_learner import StyleLearner, StyleProfile
from feedback.collector import FeedbackStore, FeedbackRecord


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_pass(self):
        gate = QualityGate(GateConfig(pass_score=0.7))
        rec = CandidateRecord(
            index=1, profile="p", source_kind="h", text="t",
            score={"final_score": 0.85, "hard_fail": False},
        )
        passed, reasons = gate.check(rec)
        assert passed
        assert len(reasons) == 0

    def test_fail_low_score(self):
        gate = QualityGate(GateConfig(pass_score=0.7))
        rec = CandidateRecord(
            index=1, profile="p", source_kind="h", text="t",
            score={"final_score": 0.5, "hard_fail": False},
        )
        passed, reasons = gate.check(rec)
        assert not passed
        assert any("score" in r for r in reasons)

    def test_fail_hard_fail(self):
        gate = QualityGate(GateConfig(pass_score=0.5))
        rec = CandidateRecord(
            index=1, profile="p", source_kind="h", text="t",
            score={"final_score": 0.9, "hard_fail": True},
        )
        passed, _ = gate.check(rec)
        assert not passed

    def test_should_continue_max_rounds(self):
        gate = QualityGate(GateConfig(max_rounds=3))
        cont, reason = gate.should_continue(3, 0.5, 0.4)
        assert not cont
        assert "max_rounds" in reason

    def test_should_continue_early_stop(self):
        gate = QualityGate(GateConfig(early_stop_score=0.9))
        cont, reason = gate.should_continue(1, 0.95, 0.0)
        assert not cont
        assert "early_stop" in reason

    def test_should_continue_no_improvement(self):
        gate = QualityGate(GateConfig(improvement_floor=0.01))
        cont, reason = gate.should_continue(2, 0.501, 0.5)
        assert not cont
        assert "improvement" in reason


# ---------------------------------------------------------------------------
# StrategyState
# ---------------------------------------------------------------------------

class TestStrategyState:
    def test_profile_tracking(self):
        state = StrategyState()
        state.record_profile_result("heuristic-natural", True, 0.85)
        state.record_profile_result("heuristic-natural", True, 0.78)
        state.record_profile_result("direct-rewrite", False, 0.5)
        ranking = state.get_profile_ranking()
        assert ranking[0][0] == "heuristic-natural"
        assert ranking[0][1].win_rate == 1.0

    def test_rule_tracking(self):
        state = StrategyState()
        state.record_rule_result("非常感谢", 0.05)
        state.record_rule_result("非常感谢", 0.03)
        state.record_rule_result("非常感谢", 0.04)
        assert "非常感谢" in state.get_effective_rules()

    def test_harmful_rules(self):
        state = StrategyState()
        for _ in range(5):
            state.record_rule_result("坏规则", -0.05)
        assert "坏规则" in state.get_harmful_rules()

    def test_rules_batch(self):
        state = StrategyState()
        state.record_rules_batch(["rule1", "rule2"], score_before=0.5, score_after=0.7)
        assert state.rules["rule1"].applied_count == 1
        assert state.rules["rule2"].applied_count == 1
        assert state.rules["rule1"].total_delta == pytest.approx(0.1)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            state = StrategyState(path)
            state.record_profile_result("test", True, 0.9)
            state.record_rule_result("rule", 0.05)
            state.save()

            state2 = StrategyState(path)
            assert "test" in state2.profiles
            assert state2.profiles["test"].wins == 1
            assert "rule" in state2.rules

    def test_policy(self):
        state = StrategyState()
        state.record_profile_result("best", True, 0.9)
        state.record_profile_result("worst", False, 0.3)
        policy = state.get_policy("default")
        assert policy["preferred_profile"] == "best"

    def test_scenario_history(self):
        state = StrategyState()
        state.record_scenario("email", "test task", 0.85, True, "heuristic", 2)
        assert len(state.scenario_history) == 1
        assert state.scenario_history[0]["scenario"] == "email"


# ---------------------------------------------------------------------------
# ProfileStats / RuleStats
# ---------------------------------------------------------------------------

class TestProfileStats:
    def test_win_rate(self):
        ps = ProfileStats(wins=3, losses=1)
        assert ps.win_rate == 0.75

    def test_win_rate_empty(self):
        ps = ProfileStats()
        assert ps.win_rate == 0.5

    def test_avg_score(self):
        ps = ProfileStats(total_score=2.5, uses=5)
        assert ps.avg_score == 0.5

    def test_round_trip(self):
        ps = ProfileStats(wins=2, losses=1, total_score=2.1, uses=3)
        d = ps.as_dict()
        ps2 = ProfileStats.from_dict(d)
        assert ps2.wins == 2
        assert ps2.losses == 1


class TestRuleStats:
    def test_effectiveness(self):
        rs = RuleStats(applied_count=4, total_delta=0.2)
        assert rs.effectiveness == 0.05

    def test_improvement_rate(self):
        rs = RuleStats(applied_count=10, improved_count=7, worsened_count=2)
        assert rs.improvement_rate == 0.7

    def test_record(self):
        rs = RuleStats()
        rs.record(0.05)
        rs.record(-0.02)
        rs.record(0.0)
        assert rs.applied_count == 3
        assert rs.improved_count == 1
        assert rs.worsened_count == 1


# ---------------------------------------------------------------------------
# StyleLearner
# ---------------------------------------------------------------------------

class TestStyleLearner:
    def test_learn_from_text(self):
        learner = StyleLearner()
        text = "你好啊。今天天气不错！你要不要出去走走？"
        profile = learner.learn_from_text(text)
        assert profile.sample_count == 1
        assert profile.avg_sentence_length > 0

    def test_learn_from_multiple(self):
        learner = StyleLearner()
        texts = [
            "简短消息。就这样。",
            "也是简短的。好的。",
        ]
        profile = learner.learn_from_texts(texts)
        assert profile.sample_count == 2

    def test_style_hints(self):
        learner = StyleLearner()
        learner.learn_from_text("太棒了！真的超开心！好激动！")
        hints = learner.get_style_hints()
        assert "target_sentence_length" in hints

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "style.json"
            learner = StyleLearner(path)
            learner.learn_from_text("测试文本。另一句话。")
            learner.save()

            learner2 = StyleLearner(path)
            assert learner2.profile.sample_count == 1


# ---------------------------------------------------------------------------
# FeedbackStore
# ---------------------------------------------------------------------------

class TestFeedbackStore:
    def test_add_and_query(self):
        store = FeedbackStore()
        store.add(FeedbackRecord(
            task="test", scenario="wechat", source_text="src",
            output_text="out", final_score=0.8, rating="good",
        ))
        store.add(FeedbackRecord(
            task="test2", scenario="wechat", source_text="src2",
            output_text="out2", final_score=0.4, rating="bad",
        ))
        assert store.total_count == 2
        assert len(store.get_good_examples()) == 1
        assert len(store.get_bad_examples()) == 1

    def test_scenario_summary(self):
        store = FeedbackStore()
        for i in range(5):
            store.add(FeedbackRecord(
                task=f"t{i}", scenario="email", source_text="s",
                output_text="o", final_score=0.7 + i * 0.02,
                rating="good" if i < 3 else "bad",
            ))
        summary = store.scenario_summary("email")
        assert summary["count"] == 5
        assert summary["good"] == 3
        assert summary["bad"] == 2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.json"
            store = FeedbackStore(path)
            store.add(FeedbackRecord(
                task="t", scenario="s", source_text="src",
                output_text="out", final_score=0.8, rating="good",
            ))

            store2 = FeedbackStore(path)
            assert store2.total_count == 1

    def test_manual_edits(self):
        store = FeedbackStore()
        store.add(FeedbackRecord(
            task="t", scenario="s", source_text="src",
            output_text="out", final_score=0.5,
            manual_edit="用户修改后的版本",
        ))
        edits = store.get_manual_edits()
        assert len(edits) == 1
        assert edits[0].manual_edit == "用户修改后的版本"
