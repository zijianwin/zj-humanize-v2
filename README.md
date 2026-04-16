# ZJ Humanize v2

A Chinese copy humanization engine for rewriting AI-sounding text into natural, human communication.

`zj-humanize-v2` 是一个面向中文场景的文案去 AI 味引擎，用来把 AI 生成、模板化、客服腔、公告腔明显的文本，改写成更自然、更像真人会直接发出去的表达。

它不是一个单纯的“润色器”，而是一条完整的 humanization pipeline。目标不是把句子写得更华丽，而是让文本从“像 AI 生成的”变成“像真人刚写的”。

## Why This Project

很多 AI 文案不是不能用，而是“太像 AI”：

- 太模板
- 太平
- 太像客服
- 太像公告
- 太礼貌
- 太像被过度润色过

这些文本通常语法没问题，信息也没错，但缺少真实沟通里该有的自然感、场景感和人味。

`zj-humanize-v2` 解决的不是基础语法问题，而是中文文本里的“人味问题”。

## What It Does

这个项目围绕中文文案 humanization 构建了一整套可执行流程，包括：

- heuristic rewrite engine
- template phrase detection
- multi-candidate generation
- scenario-adaptive scoring
- quality gate with iterative repair
- strategy evolution across runs
- style learning from feedback

它的核心目标很明确：

- 保留事实和关键信息
- 消除模板腔、客服腔和 AI 润色感
- 让口吻更符合具体沟通场景
- 输出用户真的敢直接复制发送的版本

## Typical Use Cases

适用场景包括但不限于：

- WeChat replies
- customer support messages
- email rewrites
- after-sales communication
- interview follow-ups
- manager updates
- product messaging
- community announcements
- self-media copy
- 各类“AI 写得没错，但不像人写的”中文文本

## Core Pipeline

项目当前的处理链路包括：

1. 场景检测：自动判断 `email`、`wechat`、`longform`、`service` 等场景
2. 权重适配：根据场景动态调整评分维度权重
3. 多候选生成：同时生成不同风格和来源的候选版本
4. 综合评分：从模板感、信息保留、长度、场景匹配等维度打分
5. 质量门控：不达标时自动进入修复轮次
6. 策略演化：跟踪更有效的规则与方案
7. 风格学习：从反馈和编辑中积累可复用的风格偏好

## Key Scoring Dimensions

系统会综合评估以下维度：

- `template_tone`
- `source_template_reduction`
- `must_include`
- `banned_phrases`
- `rewrite_similarity`
- `rewrite_coverage`
- `audience_fit`
- `task_facts`
- `sentence_splice`
- `length`
- `detailfulness`
- `email_shape`
- `formatting`
- `placeholder_output`
- `anti_repetition`

## Design Philosophy

这个项目遵循的原则很简单：

- Keep the facts
- Reduce template tone
- Match the real communication scene
- Prefer sendable output over decorative writing
- Keep the system modular and explainable

换成更直白的话就是：

- 不追求漂亮废话，优先像人话
- 不为了去 AI 味而牺牲事实
- 不同场景必须有不同口吻
- 用户应该敢直接把结果发出去
- 规则、评分、候选、修复流程尽量可解释、可扩展

## Installation

Requirements:

- Python 3.10+
- PyYAML

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Rewrite from a full prompt

```bash
python humanize.py --text "改写这段微信消息，去掉AI味。必须保留'退款'和'3个工作日'，不要超过120字。原文：尊敬的客户您好！非常感谢您的耐心等待..."
```

### Run from a brief file

```bash
python humanize.py examples/brief_wechat.json --verbose
```

Example brief:

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

## CLI

```bash
python humanize.py --text "用户请求"
python humanize.py brief.json
python humanize.py brief.json --verbose
python humanize.py brief.json --format json
python humanize.py --status
```

## Project Structure

```text
humanize-v2/
├── humanize.py              # main entrypoint
├── heuristics/              # rewrite rules and template detection
├── scoring/                 # multi-dimensional scoring
├── candidates/              # candidate generation and pooling
├── scripts/                 # quality gate, strategy state, style learning
├── feedback/                # feedback collection
├── reporting/               # output rendering
├── references/              # project notes and references
├── examples/                # example briefs
└── tests/                   # test suite
```

## About ZJ

这个仓库中的开发者代称统一使用 `ZJ`。

如果你想进一步理解这个项目背后的设计目标、维护方向和方法论，而不只是会调用它，可以查看：

- `references/zj_developer_notes.md`

## Current Status

当前版本已经具备：

- 可运行的 CLI
- 模块化的重写与评分架构
- 多候选 + 质量门控流程
- 基础测试覆盖
- 可继续扩展的中文场景框架

## License

暂未添加许可证。若准备开源分发，建议补充 `MIT` 或 `Apache-2.0`。
