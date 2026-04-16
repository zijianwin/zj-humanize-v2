"""Tests for candidate pool and generator."""

from __future__ import annotations

import pytest

from candidates.pool import CandidatePool, CandidateRecord
from candidates.generator import CandidateGenerator, _extract_tags
from scoring.scorer import Scorer
from heuristics.engine import ReplacementEngine


class TestCandidateRecord:
    def test_rank_key_no_hard_fail(self):
        rec = CandidateRecord(
            index=1, profile="test", source_kind="model_direct",
            text="text", score={"final_score": 0.8, "model_score": 0.7, "hard_fail": False},
        )
        key = rec.rank_key()
        assert key == (1, 0.8, 0.7)

    def test_rank_key_hard_fail(self):
        rec = CandidateRecord(
            index=1, profile="test", source_kind="model_direct",
            text="text", score={"final_score": 0.9, "hard_fail": True},
        )
        key = rec.rank_key()
        assert key[0] == 0  # hard fail penalized

    def test_as_dict(self):
        rec = CandidateRecord(
            index=1, profile="p", source_kind="heuristic", text="t",
            score={"final_score": 0.5}, failure_tags=["tag1"],
        )
        d = rec.as_dict()
        assert d["candidate_index"] == 1
        assert d["profile"] == "p"
        assert d["failure_tags"] == ["tag1"]


class TestCandidatePool:
    def test_add_and_count(self):
        pool = CandidatePool()
        pool.add(profile="a", source_kind="heuristic", text="text1", score={"final_score": 0.5})
        pool.add(profile="b", source_kind="model_direct", text="text2", score={"final_score": 0.7})
        assert pool.count == 2

    def test_pick_best(self):
        pool = CandidatePool()
        pool.add(profile="low", source_kind="h", text="t1", score={"final_score": 0.3})
        pool.add(profile="high", source_kind="h", text="t2", score={"final_score": 0.9})
        pool.add(profile="mid", source_kind="h", text="t3", score={"final_score": 0.6})
        best = pool.pick_best()
        assert best.profile == "high"

    def test_pick_best_prefers_no_hard_fail(self):
        pool = CandidatePool()
        pool.add(profile="fail", source_kind="h", text="t1",
                 score={"final_score": 0.95, "hard_fail": True})
        pool.add(profile="pass", source_kind="h", text="t2",
                 score={"final_score": 0.6, "hard_fail": False})
        best = pool.pick_best()
        assert best.profile == "pass"

    def test_pick_best_no_candidates_raises(self):
        pool = CandidatePool()
        with pytest.raises(RuntimeError):
            pool.pick_best()

    def test_pick_best_only_hard_fail(self):
        pool = CandidatePool()
        pool.add(profile="a", source_kind="h", text="t",
                 score={"final_score": 0.4, "hard_fail": True})
        best = pool.pick_best()
        assert best.profile == "a"

    def test_all_dicts(self):
        pool = CandidatePool()
        pool.add(profile="a", source_kind="h", text="t1", score={"final_score": 0.5})
        pool.add(profile="b", source_kind="h", text="t2", score={"final_score": 0.7})
        dicts = pool.all_dicts()
        assert len(dicts) == 2
        assert all("candidate_index" in d for d in dicts)


class TestExtractTags:
    def test_missing_must_include(self):
        tags = _extract_tags("text", {"notes": ["missing must_include: 退款"]}, 0.5)
        assert "missing_must_include" in tags

    def test_template_tone(self):
        tags = _extract_tags("text", {"notes": ["contains template phrases: X"]}, 0.5)
        assert "template_tone" in tags

    def test_too_similar(self):
        tags = _extract_tags("text", {"notes": ["rewrite too similar to source"]}, 0.5)
        assert "too_similar" in tags

    def test_hard_fail(self):
        tags = _extract_tags("text", {"hard_fail": True, "notes": []}, 0.5)
        assert "hard_fail" in tags

    def test_no_improvement(self):
        tags = _extract_tags("text", {"final_score": 0.3, "notes": []}, 0.5)
        assert "no_improvement" in tags

    def test_deduplication(self):
        tags = _extract_tags("text", {
            "hard_fail": True,
            "final_score": 0.1,
            "notes": ["missing must_include: X", "missing must_include: Y"],
        }, 0.5)
        assert tags.count("missing_must_include") == 1


class TestCandidateGenerator:
    def test_generate_heuristic_only(self):
        scorer = Scorer()
        engine = ReplacementEngine()
        gen = CandidateGenerator(scorer=scorer, engine=engine, llm_caller=None)
        pool = gen.generate_round(
            spec={"task": "测试"},
            task="测试",
            source_text="这是原始文本，比较长一些的文本内容用来测试。",
            current_best_text="这是原始文本，比较长一些的文本内容用来测试。",
            current_best_score=0.0,
            revision_mode="rewrite",
            failure_tags=[],
            scenario="default",
        )
        # Heuristic candidates only (may produce 0 if no rules match)
        assert isinstance(pool, CandidatePool)

    def test_generate_repair_fallback_variants(self):
        scorer = Scorer()
        engine = ReplacementEngine()
        gen = CandidateGenerator(scorer=scorer, engine=engine, llm_caller=None)
        pool = gen.generate_round(
            spec={"task": "改写微信退款回复"},
            task="改写微信退款回复",
            source_text="你好，久等了，退款这边已经在处理了，预计3个工作日内完成审核，有进展我会及时跟你说。",
            current_best_text="你好，久等了，退款这边已经在处理了，预计3个工作日内完成审核，有进展我会及时跟你说。",
            current_best_score=0.68,
            revision_mode="repair",
            failure_tags=["no_improvement"],
            scenario="wechat",
        )
        assert pool.count >= 1
        assert any(c.source_kind == "heuristic_repair" for c in pool.candidates)
