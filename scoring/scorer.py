"""Scenario-adaptive scoring system with multi-dimensional evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from heuristics.detector import TemplatePhraseDetector


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioWeights:
    """Per-dimension weights, adjustable per scenario."""

    length: float = 0.10
    must_include: float = 0.22
    banned_phrases: float = 0.12
    template_tone: float = 0.14
    source_template_reduction: float = 0.22
    rewrite_similarity: float = 0.14
    sentence_splice: float = 0.10
    placeholder_output: float = 0.12
    rewrite_coverage: float = 0.12
    formatting: float = 0.04
    detailfulness: float = 0.08
    email_shape: float = 0.05
    audience_fit: float = 0.10
    task_facts: float = 0.12
    anti_repetition: float = 0.03
    # Longform-specific dimensions
    negation_flip_density: float = 0.0
    word_repetition: float = 0.0
    parallel_structure_density: float = 0.0
    summary_sentence_pattern: float = 0.0
    explanatory_redundancy: float = 0.0

    def as_list(self) -> list[tuple[str, float]]:
        return [
            ("length", self.length),
            ("must_include", self.must_include),
            ("banned_phrases", self.banned_phrases),
            ("template_tone", self.template_tone),
            ("source_template_reduction", self.source_template_reduction),
            ("rewrite_similarity", self.rewrite_similarity),
            ("sentence_splice", self.sentence_splice),
            ("placeholder_output", self.placeholder_output),
            ("rewrite_coverage", self.rewrite_coverage),
            ("formatting", self.formatting),
            ("detailfulness", self.detailfulness),
            ("email_shape", self.email_shape),
            ("audience_fit", self.audience_fit),
            ("task_facts", self.task_facts),
            ("anti_repetition", self.anti_repetition),
            ("negation_flip_density", self.negation_flip_density),
            ("word_repetition", self.word_repetition),
            ("parallel_structure_density", self.parallel_structure_density),
            ("summary_sentence_pattern", self.summary_sentence_pattern),
            ("explanatory_redundancy", self.explanatory_redundancy),
        ]


# Pre-defined scenario weight profiles
SCENARIO_WEIGHTS: dict[str, ScenarioWeights] = {
    "email": ScenarioWeights(
        email_shape=0.12,
        detailfulness=0.10,
        template_tone=0.12,
        audience_fit=0.12,
    ),
    "wechat": ScenarioWeights(
        email_shape=0.0,
        detailfulness=0.04,
        template_tone=0.16,
        sentence_splice=0.12,
    ),
    "longform": ScenarioWeights(
        template_tone=0.16,
        source_template_reduction=0.12,
        rewrite_similarity=0.0,
        rewrite_coverage=0.0,
        email_shape=0.0,
        detailfulness=0.0,
        must_include=0.0,
        banned_phrases=0.06,
        length=0.0,
        sentence_splice=0.06,
        placeholder_output=0.06,
        formatting=0.02,
        audience_fit=0.04,
        task_facts=0.0,
        anti_repetition=0.06,
        # Longform-specific dimensions
        negation_flip_density=0.14,
        word_repetition=0.12,
        parallel_structure_density=0.08,
        summary_sentence_pattern=0.10,
        explanatory_redundancy=0.10,
    ),
    "service": ScenarioWeights(
        audience_fit=0.14,
        template_tone=0.12,
        task_facts=0.14,
    ),
    "default": ScenarioWeights(),
}

DEFAULT_GOAL = (
    "更像真人自然发送的中文沟通消息，减少模板腔、客服腔、公告腔和过度AI润色感。"
    "保持清楚、可信、有分寸。"
)

FORMAT_PENALTIES = [
    (re.compile(r"(?m)^\s*[-*]\s+"), 0.2, "contains bullet-list formatting"),
    (re.compile(r"(?m)^\s*\d+\.\s+"), 0.2, "contains numbered-list formatting"),
    (re.compile(r"[!！]{2,}"), 0.08, "contains repeated exclamation"),
    (re.compile(r"[?？]{2,}"), 0.06, "contains repeated question marks"),
    (re.compile(r"\n{3,}"), 0.1, "contains too many blank lines"),
]


@dataclass
class CandidateScore:
    """Full scoring result for a candidate."""

    final_score: float
    model_score: float
    rule_score: float
    hard_fail: bool
    char_count: int
    query: str
    rule_breakdown: dict[str, float]
    notes: list[str]
    template_details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 6),
            "model_score": round(self.model_score, 6),
            "rule_score": round(self.rule_score, 6),
            "hard_fail": self.hard_fail,
            "char_count": self.char_count,
            "query": self.query,
            "rule_breakdown": {k: round(v, 6) for k, v in self.rule_breakdown.items()},
            "notes": self.notes,
            "template_details": self.template_details,
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    """Scenario-adaptive multi-dimensional scorer."""

    def __init__(
        self,
        detector: TemplatePhraseDetector | None = None,
        model_bundle: dict[str, Any] | None = None,
    ):
        self.detector = detector or TemplatePhraseDetector()
        self._model_bundle = model_bundle

    # ----- public api -----

    def detect_scenario(self, spec: dict[str, Any]) -> str:
        """Infer the scenario from the spec's task field.

        Order matters: longform keywords are checked before wechat because
        '微信公众号' / '公众号文章' should be treated as longform, not wechat chat.
        """
        task = str(spec.get("task") or "").lower()
        # Longform MUST be checked before wechat — "公众号" contains "微信"
        # in common usage like "微信公众号", which is longform content.
        if any(m in task for m in (
            "自媒体", "文案", "小红书", "朋友圈", "口播", "种草",
            "公众号", "长文", "文章", "博客", "专栏", "推文",
            "检查ai味", "去ai味", "诊断", "长文案",
        )):
            return "longform"
        if "邮件" in task or "email" in task:
            return "email"
        if "微信" in task or "wechat" in task or "飞书" in task:
            return "wechat"
        if any(m in task for m in ("售后", "客服", "投诉", "退款", "退换")):
            return "service"
        return "default"

    def get_weights(self, scenario: str) -> ScenarioWeights:
        return SCENARIO_WEIGHTS.get(scenario, SCENARIO_WEIGHTS["default"])

    def score(
        self,
        spec: dict[str, Any],
        candidate: str,
        source_text: str = "",
        scenario: str | None = None,
    ) -> CandidateScore:
        """Full scoring pipeline."""
        candidate = candidate.strip()
        source_text = source_text.strip()
        notes: list[str] = []

        if scenario is None:
            scenario = self.detect_scenario(spec)
        weights = self.get_weights(scenario)
        task = str(spec.get("task") or "")
        query = self._build_query(spec, source_text)
        char_count = _compact_char_count(candidate)

        # Model score
        m_score = self._model_score(query, candidate)

        # Individual rule scores
        l_score = self._length_score(spec, char_count, notes)
        keep_score, missing_mi = self._must_include_score(spec, candidate, notes)
        banned_score = self._banned_phrase_score(spec, candidate, notes)

        # Template tone with severity-aware detector
        task_context = task
        detection = self.detector.detect(candidate, context=task_context)
        template_score = max(0.0, 1.0 - detection.total_penalty)
        if detection.hits:
            hit_phrases = [h.phrase for h in detection.hits[:8]]
            notes.append("contains template phrases: " + " / ".join(hit_phrases))
        template_details = detection.details

        src_reduction = self.detector.source_reduction_score(source_text, candidate, context=task_context)
        if src_reduction < 1.0:
            carried = [p.phrase for p in self.detector.phrases if p.phrase in source_text and p.phrase in candidate]
            if carried:
                notes.append("retains source template phrases: " + " / ".join(carried[:5]))

        similarity = self._rewrite_similarity(source_text, candidate, notes)
        splice = self._sentence_splice(candidate, notes)
        placeholder = self._placeholder_score(candidate, notes)
        coverage = self._rewrite_coverage(source_text, candidate, notes)
        fmt = self._formatting_score(candidate, notes)
        repeat = self._repeated_ngram(candidate, notes)
        detail = self._detail_score(spec, char_count, candidate, notes)
        email = self._email_shape_score(spec, char_count, candidate, notes)
        audience = self._audience_fit(spec, candidate, notes)
        facts = self._task_fact_score(spec, candidate, notes)

        # Longform-specific dimensions
        neg_flip = self._negation_flip_density(candidate, notes)
        word_rep = self._word_repetition_score(candidate, notes)
        parallel = self._parallel_structure_density(candidate, notes)
        summary_pat = self._summary_sentence_pattern(candidate, notes)
        explain_red = self._explanatory_redundancy(candidate, notes)

        # Weighted average
        parts = [
            ("length", l_score, weights.length),
            ("must_include", keep_score, weights.must_include),
            ("banned_phrases", banned_score, weights.banned_phrases),
            ("template_tone", template_score, weights.template_tone),
            ("source_template_reduction", src_reduction, weights.source_template_reduction),
            ("rewrite_similarity", similarity, weights.rewrite_similarity),
            ("sentence_splice", splice, weights.sentence_splice),
            ("placeholder_output", placeholder, weights.placeholder_output),
            ("rewrite_coverage", coverage, weights.rewrite_coverage),
            ("formatting", fmt, weights.formatting),
            ("detailfulness", detail, weights.detailfulness),
            ("email_shape", email, weights.email_shape),
            ("audience_fit", audience, weights.audience_fit),
            ("task_facts", facts, weights.task_facts),
            ("anti_repetition", repeat, weights.anti_repetition),
            ("negation_flip_density", neg_flip, weights.negation_flip_density),
            ("word_repetition", word_rep, weights.word_repetition),
            ("parallel_structure_density", parallel, weights.parallel_structure_density),
            ("summary_sentence_pattern", summary_pat, weights.summary_sentence_pattern),
            ("explanatory_redundancy", explain_red, weights.explanatory_redundancy),
        ]
        r_score, breakdown = _weighted_average(parts)

        final = 0.64 * m_score + 0.36 * r_score

        # Hard fail determination
        severe_template = src_reduction <= 0.05 and template_score <= 0.75
        severe_similarity = similarity <= 0.35
        severe_splice = splice <= 0.0
        severe_placeholder = placeholder <= 0.0
        severe_coverage = coverage <= 0.35

        hard_fail = (
            missing_mi
            or banned_score <= 0.36
            or l_score <= 0.2
            or detail <= 0.3
            or email <= 0.32
            or facts < 0.55
            or audience < 0.65
            or severe_template
            or severe_similarity
            or severe_splice
            or severe_placeholder
            or severe_coverage
        )

        if hard_fail:
            final *= 0.82

        return CandidateScore(
            final_score=max(0.0, min(1.0, final)),
            model_score=max(0.0, min(1.0, m_score)),
            rule_score=max(0.0, min(1.0, r_score)),
            hard_fail=hard_fail,
            char_count=char_count,
            query=query,
            rule_breakdown=breakdown,
            notes=notes,
            template_details=template_details,
        )

    # ----- internal scoring functions -----

    def _build_query(self, spec: dict[str, Any], source_text: str) -> str:
        task = str(spec.get("task", "")).strip()
        goal = _clean(str(spec.get("goal", "") or "")) or DEFAULT_GOAL
        parts = ["请判断这条候选中文沟通消息，是否更符合目标中的真人感和自然度要求。"]
        if task:
            parts.append(f"沟通任务：{task}")
        if source_text:
            parts.append(f"原始信息：{_clean(source_text)[:280]}")
        if goal:
            parts.append(f"优化目标：{goal}")
        parts.append("通用要求：像真人会发出的中文消息，避免模板腔、客服腔、公告腔和过度AI润色感。")
        notes = spec.get("style_notes") or []
        if notes:
            parts.append("风格备注：" + "；".join(_clean(str(x)) for x in notes))
        hc = spec.get("hard_constraints") or {}
        mi = hc.get("must_include") or []
        if mi:
            parts.append("必须保留：" + "；".join(_clean(str(x)) for x in mi))
        banned = hc.get("banned_phrases") or []
        if banned:
            parts.append("尽量避免：" + "；".join(_clean(str(x)) for x in banned))
        return "\n".join(parts)

    def _model_score(self, query: str, candidate: str) -> float:
        if not self._model_bundle:
            return 0.5  # neutral fallback when no model loaded
        import torch
        tokenizer = self._model_bundle["tokenizer"]
        model = self._model_bundle["model"]
        device = self._model_bundle["device"]
        inputs = tokenizer(
            [query], [candidate],
            padding=True, truncation=True, max_length=1024,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float().cpu().tolist()
        return _sigmoid(logits[0])

    def _length_score(self, spec: dict[str, Any], char_count: int, notes: list[str]) -> float:
        hc = spec.get("hard_constraints") or {}
        mn, mx = hc.get("min_chars"), hc.get("max_chars")
        if mn is None and mx is None:
            return 1.0
        score = 1.0
        if mn is not None and char_count < int(mn):
            gap = int(mn) - char_count
            score -= min(0.7, gap / max(int(mn), 1))
            notes.append(f"shorter than min_chars ({char_count} < {mn})")
        if mx is not None and char_count > int(mx):
            gap = char_count - int(mx)
            score -= min(0.85, gap / max(int(mx), 1))
            notes.append(f"longer than max_chars ({char_count} > {mx})")
        return max(0.0, score)

    def _must_include_score(self, spec: dict[str, Any], candidate: str, notes: list[str]) -> tuple[float, bool]:
        hc = spec.get("hard_constraints") or {}
        mi = [_clean(str(x)) for x in hc.get("must_include") or [] if str(x).strip()]
        if not mi:
            return 1.0, False
        hits = sum(1 for item in mi if item in candidate)
        missing = [item for item in mi if item not in candidate]
        if missing:
            notes.append("missing must_include: " + " / ".join(missing))
        return hits / len(mi), bool(missing)

    def _banned_phrase_score(self, spec: dict[str, Any], candidate: str, notes: list[str]) -> float:
        hc = spec.get("hard_constraints") or {}
        banned = [_clean(str(x)) for x in hc.get("banned_phrases") or [] if str(x).strip()]
        hits = [item for item in banned if item in candidate]
        if hits:
            notes.append("contains banned phrases: " + " / ".join(hits))
        return max(0.0, 1.0 - 0.32 * len(hits))

    def _rewrite_similarity(self, source: str, candidate: str, notes: list[str]) -> float:
        sn = _normalize_for_similarity(source)
        cn = _normalize_for_similarity(candidate)
        if len(sn) < 50 or len(cn) < 35:
            return 1.0
        ratio = SequenceMatcher(None, sn, cn).ratio()
        if ratio >= 0.9:
            notes.append(f"rewrite too similar to source (ratio={ratio:.3f})")
            return 0.0
        if ratio >= 0.84:
            notes.append(f"rewrite still very close to source (ratio={ratio:.3f})")
            return 0.35
        if ratio >= 0.78:
            notes.append(f"rewrite remains close to source (ratio={ratio:.3f})")
            return 0.68
        return 1.0

    def _sentence_splice(self, candidate: str, notes: list[str]) -> float:
        sentences = [p.strip() for p in re.split(r"[。！？\n]+", candidate) if p.strip()]
        condition = ("如有需要", "如有任何问题", "如后续", "如果后续", "后续如有")
        contact = ("欢迎随时联系", "欢迎联系", "随时联系我", "联系我")
        for s in sentences:
            c = re.sub(r"\s+", "", s)
            if re.search(r"如[^，。！？]{0,24}，如", c):
                notes.append("sentence splice issue: repeated lead-in connectors")
                return 0.0
            if re.search(r"稍后前(?=给|向|为|把|会|同步|回复)", c):
                notes.append("sentence splice issue: invalid time connector")
                return 0.0
            hc = sum(1 for t in condition if t in c)
            hk = sum(1 for t in contact if t in c)
            if hc >= 2 or (hc >= 1 and hk >= 1 and "，" in s):
                notes.append("sentence splice issue: collided closing phrases")
                return 0.0
        return 1.0

    def _placeholder_score(self, candidate: str, notes: list[str]) -> float:
        c = re.sub(r"\s+", "", candidate)
        if not c:
            notes.append("candidate is empty")
            return 0.0
        if re.search(r"(……|\.{3,}|。。。)", c):
            notes.append("contains placeholder-style ellipsis")
            return 0.0
        if c in {"同上", "略", "待补充", "待确认"}:
            notes.append("contains placeholder-style content")
            return 0.0
        return 1.0

    def _rewrite_coverage(self, source: str, candidate: str, notes: list[str]) -> float:
        sc = _compact_char_count(source)
        cc = _compact_char_count(candidate)
        if sc < 60:
            return 1.0
        ratio = cc / max(sc, 1)
        if ratio < 0.15:
            notes.append(f"rewrite drops too much source detail (ratio={ratio:.3f})")
            return 0.0
        if ratio < 0.24:
            notes.append(f"rewrite is over-compressed (ratio={ratio:.3f})")
            return 0.35
        if ratio < 0.32:
            notes.append(f"rewrite is quite compressed (ratio={ratio:.3f})")
            return 0.72
        return 1.0

    def _formatting_score(self, candidate: str, notes: list[str]) -> float:
        score = 1.0
        for pat, pen, msg in FORMAT_PENALTIES:
            if pat.search(candidate):
                score -= pen
                notes.append(msg)
        return max(0.0, score)

    def _repeated_ngram(self, candidate: str, notes: list[str]) -> float:
        c = re.sub(r"\s+", "", candidate)
        if len(c) < 10:
            return 1.0
        seen: dict[str, int] = {}
        for i in range(max(0, len(c) - 3)):
            g = c[i:i + 4]
            seen[g] = seen.get(g, 0) + 1
        repeated = [g for g, cnt in seen.items() if cnt >= 3]
        if repeated:
            notes.append("repeated 4-gram fragments: " + " / ".join(repeated[:5]))
        return max(0.0, 1.0 - min(0.25, 0.06 * len(repeated)))

    def _detail_score(self, spec: dict[str, Any], char_count: int, candidate: str, notes: list[str]) -> float:
        task = str(spec.get("task") or "")
        has_progress = any(t in candidate for t in ("正在", "已经", "这边", "目前", "预计", "会", "给您", "回复"))
        if char_count < 10:
            notes.append("too short for a believable reply")
            return 0.28
        if char_count < 14:
            notes.append("too short and vague for a complete reply")
            return 0.5
        if any(t in task for t in ("客户", "上级", "老板", "面试", "售后")) and char_count < 18:
            notes.append("light on concrete progress detail")
            return 0.72 if has_progress else 0.62
        if not has_progress and char_count < 22:
            notes.append("could use clearer action or time detail")
            return 0.82
        return 1.0

    def _email_shape_score(self, spec: dict[str, Any], char_count: int, candidate: str, notes: list[str]) -> float:
        task = str(spec.get("task") or "")
        if "邮件" not in task and "email" not in task.lower():
            return 1.0
        hc = spec.get("hard_constraints") or {}
        mx = hc.get("max_chars")
        tight = mx is not None and int(mx) <= 140
        score = 1.0
        has_greeting = any(t in candidate for t in ("您好", "尊敬", "Hi", "Hello", "Dear"))
        sents = len([p for p in re.split(r"[。！？!?]\s*", candidate) if p.strip()])
        if char_count < (32 if tight else 45):
            notes.append("too short for an email reply")
            score -= 0.55
        if not has_greeting:
            notes.append("missing email-style greeting")
            score -= 0.32
        if sents < 2 and not tight:
            notes.append("reads like a chat reply, not a full email body")
            score -= 0.38
        return max(0.0, score)

    def _audience_fit(self, spec: dict[str, Any], candidate: str, notes: list[str]) -> float:
        task = str(spec.get("task") or "")
        score = 1.0
        if "客户" in task:
            if any(t in candidate for t in ("XX总", "x总", "老板", "总，您好")):
                notes.append("wrong audience: reads like a manager-facing salutation")
                score -= 0.55
        if any(t in task for t in ("老板", "上级", "经理")):
            if "尊敬的客户" in candidate:
                notes.append("wrong audience: reads like a customer-facing reply")
                score -= 0.55
        return max(0.0, score)

    def _task_fact_score(self, spec: dict[str, Any], candidate: str, notes: list[str]) -> float:
        task = str(spec.get("task") or "")
        if not task:
            return 1.0
        checks: list[tuple[str, tuple[str, ...]]] = []
        for m in re.finditer(
            r"(周[一二三四五六日天](?:早上|上午|中午|下午|晚上)?|明天(?:早上|上午|中午|下午|晚上)?|明早|明晚|今天(?:早上|上午|中午|下午|晚上)?|今晚|本周内)",
            task,
        ):
            checks.append((m.group(1), (m.group(1),)))
        fact_groups = [
            ("空调", ("空调",)), ("退款", ("退款", "退")), ("财务", ("财务",)),
            ("面试", ("面试",)), ("合同", ("合同",)), ("报价", ("报价", "价格")),
            ("维修", ("维修", "修")), ("破损", ("破损", "损坏", "坏")),
        ]
        for marker, alts in fact_groups:
            if marker in task:
                checks.append((marker, alts))
        if not checks:
            return 1.0
        missing: list[str] = []
        hits = 0
        seen: set[str] = set()
        for label, alts in checks:
            if label in seen:
                continue
            seen.add(label)
            if any(v in candidate for v in alts):
                hits += 1
            else:
                missing.append(label)
        if missing:
            notes.append("missing task facts: " + " / ".join(missing))
        return hits / max(len(seen), 1)

    # ----- longform-specific scoring functions -----

    def _negation_flip_density(self, candidate: str, notes: list[str]) -> float:
        """Detect overuse of 'not A but B' (不是A而是B) structures."""
        patterns = [
            r"不是.{1,20}而是",
            r"不在于.{1,20}而在于",
            r"不是.{1,20}是(?!否)",
            r"并非.{1,20}而是",
            r"不是因为.{1,20}是因为",
        ]
        total_hits = 0
        for pat in patterns:
            total_hits += len(re.findall(pat, candidate))
        char_count = _compact_char_count(candidate)
        if char_count < 200:
            return 1.0
        # Density per 1000 chars
        density = total_hits / (char_count / 1000)
        if density >= 5.0:
            notes.append(f"negation-flip overuse: {total_hits} instances (density={density:.1f}/1k chars)")
            return 0.2
        if density >= 3.5:
            notes.append(f"negation-flip high density: {total_hits} instances (density={density:.1f}/1k chars)")
            return 0.5
        if density >= 2.0:
            notes.append(f"negation-flip moderate density: {total_hits} instances (density={density:.1f}/1k chars)")
            return 0.75
        return 1.0

    def _word_repetition_score(self, candidate: str, notes: list[str]) -> float:
        """Detect repeated keywords within close proximity (sliding window)."""
        # Split into sentences
        sentences = [s.strip() for s in re.split(r"[。！？\n]+", candidate) if s.strip() and len(s.strip()) > 4]
        if len(sentences) < 3:
            return 1.0

        # Extract content words (2+ chars, excluding common particles)
        stop_words = {"一个", "这个", "那个", "什么", "为什么", "怎么", "可以", "已经",
                      "因为", "所以", "但是", "而且", "如果", "就是", "不是", "没有",
                      "他们", "自己", "我们", "你们", "可能", "还是", "这样", "那样",
                      "其实", "现在", "过去", "真正", "表面", "一种", "一样"}
        repeated_pairs: list[str] = []
        window_size = 4  # check within 4 consecutive sentences

        for i in range(max(0, len(sentences) - window_size + 1)):
            window = sentences[i:i + window_size]
            word_counts: dict[str, int] = {}
            for sent in window:
                words = re.findall(r"[\u4e00-\u9fff]{2,4}", sent)
                seen_in_sent: set[str] = set()
                for w in words:
                    if w not in stop_words and w not in seen_in_sent:
                        seen_in_sent.add(w)
                        word_counts[w] = word_counts.get(w, 0) + 1
            for w, cnt in word_counts.items():
                if cnt >= 3 and w not in [p for p in repeated_pairs]:
                    repeated_pairs.append(w)

        if not repeated_pairs:
            return 1.0
        notes.append(f"word repetition in close proximity: {' / '.join(repeated_pairs[:5])}")
        return max(0.3, 1.0 - 0.15 * len(repeated_pairs))

    def _parallel_structure_density(self, candidate: str, notes: list[str]) -> float:
        """Detect overly dense parallel/排比 sentence structures."""
        sentences = [s.strip() for s in re.split(r"[。！？\n]+", candidate) if s.strip() and len(s.strip()) > 6]
        if len(sentences) < 5:
            return 1.0

        # Check for consecutive sentences with same opening pattern
        consecutive_same = 0
        max_consecutive = 0
        for i in range(1, len(sentences)):
            # Compare first 2 chars (common Chinese parallel pattern)
            if len(sentences[i]) >= 2 and len(sentences[i - 1]) >= 2:
                if sentences[i][:2] == sentences[i - 1][:2]:
                    consecutive_same += 1
                    max_consecutive = max(max_consecutive, consecutive_same)
                else:
                    consecutive_same = 0

        # Also check "他/她+verb" pattern
        pronoun_start = 0
        for s in sentences:
            if re.match(r"^[他她它](?:的|是|在|不|从|把|被|让|给|也|又|却|就|还)", s):
                pronoun_start += 1

        total_parallel = max_consecutive
        char_count = _compact_char_count(candidate)
        if char_count < 300:
            return 1.0

        if total_parallel >= 4:
            notes.append(f"excessive parallel structure: {total_parallel + 1} consecutive same-pattern sentences")
            return 0.4
        if total_parallel >= 3:
            notes.append(f"high parallel structure density: {total_parallel + 1} consecutive same-pattern sentences")
            return 0.65
        if pronoun_start > len(sentences) * 0.4:
            notes.append(f"repetitive pronoun-start sentences: {pronoun_start}/{len(sentences)}")
            return 0.7
        return 1.0

    def _summary_sentence_pattern(self, candidate: str, notes: list[str]) -> float:
        """Detect AI-style summary/金句 sentence patterns."""
        patterns = [
            r"从来不是.{2,15}而是.{2,15}",
            r"真正的.{2,10}不是.{2,15}而是",
            r"不是为了.{2,15}而是为了",
            r"本质上.{2,10}不是.{2,15}而是",
            r".{2,6}的.{2,6}，从来不是",
        ]
        total_hits = 0
        hit_details: list[str] = []
        for pat in patterns:
            matches = re.findall(pat, candidate)
            if matches:
                total_hits += len(matches)
                for m in matches[:2]:
                    hit_details.append(m[:30])

        char_count = _compact_char_count(candidate)
        if char_count < 300:
            return 1.0

        density = total_hits / (char_count / 1000)
        if density >= 3.0:
            notes.append(f"AI-style summary sentences: {total_hits} instances — {' / '.join(hit_details)}")
            return 0.3
        if density >= 2.0:
            notes.append(f"summary sentence pattern density: {total_hits} instances")
            return 0.6
        if total_hits >= 2:
            notes.append(f"summary sentence patterns found: {total_hits}")
            return 0.8
        return 1.0

    def _explanatory_redundancy(self, candidate: str, notes: list[str]) -> float:
        """Detect adjacent sentences that express the same idea with different words."""
        # Check for patterns where a metaphor is immediately explained
        redundancy_patterns = [
            # "冰山之下" + "隐蔽/深层/底层"
            (r"冰山之下.{0,20}(?:隐蔽|深层|更深|底层)", "冰山之下 + 隐蔽/深层 redundancy"),
            # "旧地图" + "过去的方法/过去的打法"
            (r"旧地图.{0,30}(?:过去的|以前的)(?:方法|打法|路径)", "旧地图 + 过去的方法 redundancy"),
            # "壁垒" + "阻碍/障碍/门槛"
            (r"壁垒.{0,20}(?:阻碍|障碍|门槛)", "壁垒 + 阻碍 redundancy"),
            # Repeated emphasis: same concept within 2 sentences using synonyms
            (r"意义(?:感|危机).{0,60}(?:价值|目的|方向)(?:感|危机)", "meaning-related concept repetition"),
        ]
        hits = 0
        hit_msgs: list[str] = []
        for pat, msg in redundancy_patterns:
            if re.search(pat, candidate):
                hits += 1
                hit_msgs.append(msg)

        if hits == 0:
            return 1.0
        notes.append(f"explanatory redundancy: {' / '.join(hit_msgs)}")
        return max(0.4, 1.0 - 0.2 * hits)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compact_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _normalize_for_similarity(text: str) -> str:
    c = re.sub(r"\s+", "", text)
    return re.sub(r"[，。！？；：、,.!?:;\"'\u201c\u201d\u2018\u2019（）()《》【】\\\[\]<>·\u2014\u2026\-]", "", c)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _weighted_average(parts: list[tuple[str, float, float]]) -> tuple[float, dict[str, float]]:
    total_w = sum(w for _, _, w in parts if w > 0)
    if total_w <= 0:
        return 0.0, {n: s for n, s, _ in parts}
    val = sum(s * w for _, s, w in parts if w > 0) / total_w
    return val, {n: s for n, s, _ in parts}
