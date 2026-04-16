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


def test_chase_progress_message_becomes_more_direct():
    text = (
        "帮我把这段催进度微信改得更自然一点。"
        "原文：想礼貌跟进一下目前的处理进度，如您方便的话，烦请同步一下最新进度。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert "礼貌跟进" not in result.final_text
    assert "烦请" not in result.final_text
    assert "最新进度" in result.final_text


def test_after_sales_soothing_message_loses_overformal_apology():
    text = (
        "把这段售后安抚消息改得更像真人说的话。"
        "原文：给您带来不便我们深感抱歉，我们非常理解您的心情，我们会尽快为您处理，请您放心。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert "深感抱歉" not in result.final_text
    assert "请您放心" not in result.final_text
    assert "抱歉" in result.final_text or "不好意思" in result.final_text


def test_interview_followup_email_becomes_more_natural():
    text = (
        "帮我润色这段面试跟进邮件，让它更自然。"
        "原文：感谢您今天安排的面试，期待后续能够有进一步沟通的机会，期待您的进一步通知。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert "期待您的进一步通知" not in result.final_text
    assert "联系" in result.final_text or "消息" in result.final_text


def test_manager_update_becomes_less_report_like():
    text = (
        "把这段上级汇报改得更自然一点。"
        "原文：现将当前工作进展同步如下：已完成阶段性工作内容，下一步将重点推进供应商对接，如无特殊情况，本周内可以完成。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert "同步如下" not in result.final_text
    assert "下一步将重点推进" not in result.final_text
    assert "：：" not in result.final_text
    assert "接下来" in result.final_text or "目前" in result.final_text


def test_group_notice_becomes_more_human():
    text = (
        "把这段社群通知改得更自然一点。"
        "原文：感谢各位一直以来的理解与支持，现将相关安排通知如下，请各位知悉并提前做好安排，如有疑问可随时联系工作人员。"
    )
    spec, source = _parse_text_request(text)
    engine = HumanizeEngine()
    result = engine.run(spec, source)

    assert "通知如下" not in result.final_text
    assert "知悉并提前做好安排" not in result.final_text
    assert "：，" not in result.final_text
    assert "大家" in result.final_text or "群里" in result.final_text
