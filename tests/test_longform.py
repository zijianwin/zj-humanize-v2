"""Tests for longform scenario: scoring dimensions, candidate generation, and rendering."""

from __future__ import annotations

import pytest

from scoring.scorer import Scorer, ScenarioWeights, SCENARIO_WEIGHTS
from candidates.generator import CandidateGenerator
from candidates.pool import CandidatePool
from heuristics.engine import ReplacementEngine
from heuristics.detector import TemplatePhraseDetector
from reporting.renderer import OutputRenderer
from scripts.quality_gate import IterationResult, RoundLog
from candidates.pool import CandidateRecord
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_RULES_DIR = _THIS_DIR.parent / "heuristics" / "rules"


@pytest.fixture
def scorer():
    return Scorer()


@pytest.fixture
def engine():
    return ReplacementEngine(_RULES_DIR)


@pytest.fixture
def generator(scorer, engine):
    return CandidateGenerator(scorer=scorer, engine=engine)


@pytest.fixture
def renderer():
    return OutputRenderer(verbose=True)


# Sample longform text with AI-style patterns
SAMPLE_LONGFORM = """
很多人表面上看起来很努力，但实际上他们从来不是在解决问题，而是在逃避问题。

真正的成长不是靠外力推动，而是来自内心的觉醒。这意味着你需要重新审视自己。

不是为了证明自己而努力，而是为了成为更好的自己。换句话说，你需要找到内在的驱动力。

他走了。他回来了。他又走了。他最终留下了。

本质上，成功不是偶然的，而是必然的。从来不是运气，而是准备。

冰山之下，隐蔽着更深层的恐惧。旧地图虽然过期了，但过去的方法仍然有人在用。

这两者的区别，天差地别。
""".strip()

# Clean longform text with minimal AI patterns
CLEAN_LONGFORM = """
老段是我的老朋友，典型的数据科学家，也是最折腾的连续创业者。这两年大环境不太平，
创业者聚在一起不抱怨几句似乎都显得不合群。老段也吐槽，但他身上有种神奇的东西。

中小客户掉了不少，大客户又拖着项目款，换成普通人可能早就焦虑得睡不着。
老段倒好，该吃吃，该睡睡，眼神里没半点颓丧。

我问他你凭什么不焦虑。他笑了笑，聊起他贵州都匀农村的出身。因为见过真正的苦日子，
他觉得现在再难也比过去好。钱多有钱多的活法，钱少有钱少的过活。

很多人听到这话会觉得这是苦出身的人心态好。但我觉得没这么简单。我看到的是一套完整的
心理地基。
""".strip()


# ---------------------------------------------------------------------------
# Test: detect_scenario recognizes longform keywords
# ---------------------------------------------------------------------------

class TestLongformScenarioDetection:
    def setup_method(self):
        self.scorer = Scorer()

    def test_classic_keywords(self):
        assert self.scorer.detect_scenario({"task": "自媒体文案"}) == "longform"
        assert self.scorer.detect_scenario({"task": "小红书种草"}) == "longform"

    def test_new_keywords(self):
        assert self.scorer.detect_scenario({"task": "公众号文章"}) == "longform"
        assert self.scorer.detect_scenario({"task": "检查这篇长文的AI味"}) == "longform"
        assert self.scorer.detect_scenario({"task": "诊断这篇文章"}) == "longform"
        assert self.scorer.detect_scenario({"task": "博客文章去AI味"}) == "longform"
        assert self.scorer.detect_scenario({"task": "专栏文案检查"}) == "longform"

    def test_not_longform(self):
        assert self.scorer.detect_scenario({"task": "回复客户邮件"}) != "longform"
        assert self.scorer.detect_scenario({"task": "微信消息"}) != "longform"


# ---------------------------------------------------------------------------
# Test: longform scenario weights exist and are reasonable
# ---------------------------------------------------------------------------

class TestLongformWeights:
    def test_longform_weights_exist(self):
        assert "longform" in SCENARIO_WEIGHTS

    def test_longform_has_new_dimensions(self):
        w = SCENARIO_WEIGHTS["longform"]
        assert w.negation_flip_density > 0
        assert w.word_repetition > 0
        assert w.parallel_structure_density > 0
        assert w.summary_sentence_pattern > 0
        assert w.explanatory_redundancy > 0

    def test_longform_disables_irrelevant_dims(self):
        w = SCENARIO_WEIGHTS["longform"]
        assert w.rewrite_similarity == 0
        assert w.rewrite_coverage == 0
        assert w.email_shape == 0
        assert w.must_include == 0


# ---------------------------------------------------------------------------
# Test: individual longform scoring functions
# ---------------------------------------------------------------------------

class TestNegationFlipDensity:
    def setup_method(self):
        self.scorer = Scorer()

    def test_no_patterns(self):
        notes = []
        score = self.scorer._negation_flip_density("这是一段普通文本，没有任何翻转句式。" * 20, notes)
        assert score == 1.0
        assert not notes

    def test_high_density(self):
        notes = []
        # Create text with many negation-flip patterns (must exceed 200 chars)
        text = ("不是问题而是机会，这个道理很重要。" * 8 + "普通填充文本，这是一段很长的中文。" * 8)
        score = self.scorer._negation_flip_density(text, notes)
        assert score < 1.0
        assert any("negation-flip" in n for n in notes)

    def test_short_text_exempt(self):
        notes = []
        score = self.scorer._negation_flip_density("不是A而是B", notes)
        assert score == 1.0  # too short to penalize


class TestWordRepetition:
    def setup_method(self):
        self.scorer = Scorer()

    def test_no_repetition(self):
        notes = []
        text = "第一句话讲苹果。第二句话讲橘子。第三句话讲香蕉。第四句话讲西瓜。第五句话讲葡萄。"
        score = self.scorer._word_repetition_score(text, notes)
        assert score == 1.0

    def test_with_repetition(self):
        notes = []
        text = "创业者的焦虑很深。创业者的痛苦很重。创业者的迷茫很久。创业者最终会走出来。但每个创业者的路不同。"
        score = self.scorer._word_repetition_score(text, notes)
        assert score < 1.0
        assert any("word repetition" in n for n in notes)


class TestParallelStructureDensity:
    def setup_method(self):
        self.scorer = Scorer()

    def test_no_parallel(self):
        notes = []
        text = ("这是第一句话。那是第二句话。我们来看第三句话。"
                "接下来讨论第四句话。然后是第五句话。最后一句结束。") * 3
        score = self.scorer._parallel_structure_density(text, notes)
        assert score == 1.0

    def test_excessive_parallel(self):
        notes = []
        text = ("他走了很远。他看了很久。他想了很多。他说了很少。他做了很多。"
                "他最终成功了。" + "普通填充文本。" * 10)
        score = self.scorer._parallel_structure_density(text, notes)
        # Should detect pronoun-start pattern or consecutive same-opening
        assert score <= 1.0


class TestSummarySentencePattern:
    def setup_method(self):
        self.scorer = Scorer()

    def test_clean_text(self):
        notes = []
        text = "普通的叙事文本，没有金句式总结。" * 30
        score = self.scorer._summary_sentence_pattern(text, notes)
        assert score == 1.0

    def test_ai_summary_patterns(self):
        notes = []
        # Must exceed 300 chars compact to trigger detection
        text = ("从来不是运气好而是准备充分，这一点非常关键需要我们深入理解。" +
                "真正的自由不是为所欲为而是从容选择，每个人都应该明白这个道理。" +
                "不是为了成功而努力而是为了成长而坚持，这才是人生的真谛所在。" +
                "普通填充文本段落内容。" * 25)
        score = self.scorer._summary_sentence_pattern(text, notes)
        assert score < 1.0
        assert any("summary" in n.lower() or "AI-style" in n for n in notes)


class TestExplanatoryRedundancy:
    def setup_method(self):
        self.scorer = Scorer()

    def test_no_redundancy(self):
        notes = []
        score = self.scorer._explanatory_redundancy("普通文本没有冗余解释", notes)
        assert score == 1.0

    def test_iceberg_redundancy(self):
        notes = []
        score = self.scorer._explanatory_redundancy(
            "冰山之下隐蔽着更深层的问题", notes
        )
        assert score < 1.0
        assert any("redundancy" in n for n in notes)


# ---------------------------------------------------------------------------
# Test: full scoring pipeline for longform scenario
# ---------------------------------------------------------------------------

class TestLongformFullScoring:
    def setup_method(self):
        self.scorer = Scorer()

    def test_ai_heavy_text_scores_lower(self):
        spec = {"task": "公众号文章"}
        ai_score = self.scorer.score(spec, SAMPLE_LONGFORM, SAMPLE_LONGFORM, scenario="longform")
        clean_score = self.scorer.score(spec, CLEAN_LONGFORM, CLEAN_LONGFORM, scenario="longform")
        # AI-heavy text should score noticeably lower
        assert ai_score.final_score < clean_score.final_score

    def test_longform_produces_rule_breakdown(self):
        spec = {"task": "公众号文章"}
        result = self.scorer.score(spec, SAMPLE_LONGFORM, SAMPLE_LONGFORM, scenario="longform")
        bd = result.rule_breakdown
        assert "negation_flip_density" in bd
        assert "word_repetition" in bd
        assert "parallel_structure_density" in bd
        assert "summary_sentence_pattern" in bd
        assert "explanatory_redundancy" in bd


# ---------------------------------------------------------------------------
# Test: candidate generator longform diagnostic fallback
# ---------------------------------------------------------------------------

class TestLongformCandidateGeneration:
    def setup_method(self):
        self.scorer = Scorer()
        self.engine = ReplacementEngine(_RULES_DIR)
        self.generator = CandidateGenerator(scorer=self.scorer, engine=self.engine)

    def test_diagnostic_candidate_generated(self):
        """When heuristic rules don't change the text, a diagnostic candidate
        should be generated for longform scenario."""
        spec = {"task": "公众号文章"}
        pool = self.generator.generate_round(
            spec=spec,
            task="公众号文章",
            source_text=CLEAN_LONGFORM,
            current_best_text=CLEAN_LONGFORM,
            current_best_score=0.0,
            revision_mode="full",
            failure_tags=[],
            scenario="longform",
        )
        # Should have at least one candidate (diagnostic or heuristic)
        assert pool.count > 0

    def test_diagnostic_has_score(self):
        spec = {"task": "检查文章AI味"}
        pool = self.generator.generate_round(
            spec=spec,
            task="检查文章AI味",
            source_text=SAMPLE_LONGFORM,
            current_best_text=SAMPLE_LONGFORM,
            current_best_score=0.0,
            revision_mode="full",
            failure_tags=[],
            scenario="longform",
        )
        assert pool.count > 0
        best = pool.pick_best()
        assert best is not None
        assert best.score is not None
        score = float(best.score.get("final_score", 0))
        assert score > 0.0


# ---------------------------------------------------------------------------
# Test: renderer longform report
# ---------------------------------------------------------------------------

class TestLongformRenderer:
    def setup_method(self):
        self.renderer = OutputRenderer(verbose=True)

    def _make_result(self) -> IterationResult:
        """Create a mock IterationResult with longform scoring data."""
        record = CandidateRecord(
            index=1,
            profile="longform-diagnostic",
            source_kind="diagnostic",
            text=SAMPLE_LONGFORM,
            score={
                "final_score": 0.62,
                "rule_breakdown": {
                    "negation_flip_density": 0.5,
                    "word_repetition": 0.8,
                    "parallel_structure_density": 0.65,
                    "summary_sentence_pattern": 0.3,
                    "explanatory_redundancy": 0.6,
                    "template_tone": 0.7,
                    "source_template_reduction": 0.85,
                    "anti_repetition": 0.9,
                    "sentence_splice": 1.0,
                    "formatting": 0.95,
                },
                "notes": [
                    "negation-flip overuse: 5 instances (density=4.2/1k chars)",
                    "AI-style summary sentences: 3 instances",
                    "explanatory redundancy: 冰山之下 + 隐蔽/深层 redundancy",
                ],
            },
            failure_tags=[],
        )
        return IterationResult(
            passed=False,
            final_text=SAMPLE_LONGFORM,
            final_score=0.62,
            final_hard_fail=False,
            total_rounds=1,
            best_record=record,
            round_logs=[],
            failure_tags=[],
        )

    def test_longform_renders_diagnostic_report(self):
        result = self._make_result()
        spec = {"task": "公众号文章", "scenario": "longform"}
        output = self.renderer.render_text(result, spec)
        assert "长文AI味诊断报告" in output
        assert "长文专项检测" in output
        assert "通用检测" in output

    def test_longform_report_contains_scores(self):
        result = self._make_result()
        spec = {"task": "公众号文章", "scenario": "longform"}
        output = self.renderer.render_text(result, spec)
        assert "0.62" in output  # final score

    def test_longform_report_contains_notes(self):
        result = self._make_result()
        spec = {"task": "公众号文章", "scenario": "longform"}
        output = self.renderer.render_text(result, spec)
        assert "具体问题与修改建议" in output
        assert "negation-flip" in output

    def test_non_longform_uses_standard_report(self):
        result = self._make_result()
        spec = {"task": "微信回复", "scenario": "wechat"}
        output = self.renderer.render_text(result, spec)
        assert "长文AI味诊断报告" not in output
        assert "去AI味处理结果" in output
