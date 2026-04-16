---
name: zj-humanize-v2
description: "中文文案去AI味和人味优化 skill。将AI生成的中文文案改写为更自然、更像真人发送的沟通消息，消除模板腔、客服腔、公告腔和过度AI润色感。适用于自媒体文案、客户邮件、微信回复、飞书消息、售后沟通、面试跟进、上级汇报、产品宣传和社群通知。当用户提到去AI味、humanize、改写得更自然、去模板腔、去客服腔、让文案更像人写的、润色中文文案、优化文案口吻等需求时，请务必使用本 skill，即使用户没有明确提到 skill 名称。"
---

> Important: all script paths are relative to this skill directory.
> Preferred entrypoint: `cd {this_skill_dir} && python humanize.py --text "{entire_user_request}"`
> Alternative entrypoint: `cd {this_skill_dir} && python humanize.py <brief.json path> [--verbose]`
> Always set the shell tool timeout to at least `120` seconds for `python humanize.py`.
> Invocation rule: do not build helper JSON or temporary Python snippets to call this skill. Invoke `python humanize.py --text ...` directly with the user's full request.

# ZJ Humanize v2 — 去AI味 Skill

## What This Skill Does

本 skill 接收一段 AI 生成（或模板化）的中文沟通文案，通过 **启发式规则引擎 + 模型打分 + 多候选竞争 + 迭代修复** 的全链路方案，将其改写为更像真人自然发送的版本。

用户只需要提供：
- 待改写的文案（原始文本）
- 可选：沟通任务描述、字数约束、必须保留的关键信息、禁用短语

## How to Invoke

### 方式一：`--text` 直通模式（推荐）

将用户的完整请求原样传入：

```bash
cd {this_skill_dir} && python humanize.py --text "帮我把这段客服回复改得更自然：尊敬的客户您好！非常感谢您的耐心等待。关于您的退款申请，我们已经在积极处理中..."
```

如果用户同时提供了约束条件：

```bash
cd {this_skill_dir} && python humanize.py --text "改写这段微信消息，去掉AI味。必须保留'退款'和'3个工作日'，不要超过120字。原文：尊敬的客户您好！非常感谢您的耐心等待..."
```

### 方式二：Brief JSON 文件

```bash
cd {this_skill_dir} && python humanize.py brief.json --verbose
```

Brief JSON 格式：
```json
{
  "task": "回复客户关于退款进度的微信消息",
  "source_text": "尊敬的客户您好！非常感谢您的耐心等待...",
  "hard_constraints": {
    "must_include": ["退款", "3个工作日"],
    "banned_phrases": ["尊敬的"],
    "max_chars": 120
  }
}
```

## Processing Pipeline

1. **场景检测**：根据 task 关键词自动判断场景（email / wechat / longform / service）
2. **权重适配**：不同场景使用不同的评分维度权重
3. **多候选生成**：同时生成启发式（natural/balanced 两种风格）和模型候选
4. **综合评分**：15 个维度的加权评分 + 模型打分
5. **质量门控**：未达标时自动进入修复轮次（最多 4 轮）
6. **策略演化**：跨会话跟踪哪些方案和规则更有效

## Scoring Dimensions

| 维度 | 说明 |
|------|------|
| template_tone | 模板腔严重度（分 4 级） |
| source_template_reduction | 原文模板短语的消除率 |
| must_include | 必须保留的关键信息 |
| banned_phrases | 禁用短语检查 |
| rewrite_similarity | 改写与原文的差异度 |
| rewrite_coverage | 改写覆盖率 |
| audience_fit | 受众匹配度 |
| task_facts | 任务事实覆盖 |
| sentence_splice | 句式拼接问题 |
| length | 字数约束 |
| detailfulness | 具体程度 |
| email_shape | 邮件体格式 |
| formatting | 格式规范性 |
| placeholder_output | 占位符输出检测 |
| anti_repetition | 重复片段检测 |

## CLI Options

```
python humanize.py --text "用户请求"         # 直通模式
python humanize.py brief.json               # Brief 文件模式
python humanize.py brief.json --verbose      # 详细输出
python humanize.py brief.json --format json  # JSON 输出
python humanize.py --status                  # 查看系统状态
```

## Developer Notes

- 如果需要理解这个 skill 的设计目标、取舍原则和维护方向，读取 [references/zj_developer_notes.md](references/zj_developer_notes.md)。
- 该说明中的作者代称统一使用 `ZJ`。
