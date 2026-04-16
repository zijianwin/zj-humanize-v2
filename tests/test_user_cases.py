from __future__ import annotations

from humanize import HumanizeEngine, _parse_text_request


def test_parse_text_request_extracts_multiple_must_include_items():
    spec, source = _parse_text_request(
        "用 zj-humanize-v2 改写这段微信回复，保留'退款'和'3个工作日'，不要太像客服话术。原文：尊敬的客户您好！非常感谢您的耐心等待。"
    )
    assert source.startswith("尊敬的客户您好")
    assert spec["hard_constraints"]["must_include"] == ["退款", "3个工作日"]


def test_wechat_refund_reply_is_humanized_beyond_template_phrases():
    text = (
        "用 zj-humanize-v2 帮我把这段微信回复改得更自然一点，去掉 AI 味，"
        "保留'退款'和'3个工作日'，不要太像客服话术。"
        "原文：尊敬的客户您好！非常感谢您的耐心等待，关于您的退款申请，我们已经在积极处理中，"
        "预计将在3个工作日内完成审核，请您耐心关注后续通知。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert "退款" in result.final_text
    assert "3个工作日" in result.final_text
    assert "尊敬的客户您好" not in result.final_text
    assert "请您耐心关注后续通知" not in result.final_text
    assert "，！" not in result.final_text


def test_heuristic_only_mode_can_pass_quality_gate_for_good_short_reply():
    text = (
        "用 zj-humanize-v2 帮我把这段微信回复改得更自然一点，去掉 AI 味，"
        "保留'退款'和'3个工作日'，不要太像客服话术。"
        "原文：尊敬的客户您好！非常感谢您的耐心等待，关于您的退款申请，我们已经在积极处理中，"
        "预计将在3个工作日内完成审核，请您耐心关注后续通知。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert result.final_score >= 0.72
