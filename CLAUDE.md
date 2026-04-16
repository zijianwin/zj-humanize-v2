# Humanize v2 — Claude Code 使用说明

## 项目结构

```
humanize-v2/
├── humanize.py              # 主入口 + CLI
├── heuristics/              # 启发式规则引擎
│   ├── engine.py            # 替换规则引擎（精确/正则/模糊匹配）
│   ├── detector.py          # 模板短语检测器（严重度分级）
│   └── rules/               # YAML 规则文件
│       ├── template_phrases.yaml
│       ├── longform.yaml
│       ├── email.yaml
│       └── short_reply.yaml
├── scoring/                 # 评分系统
│   └── scorer.py            # 15维场景自适应评分
├── candidates/              # 候选生成
│   ├── generator.py         # 模型+启发式协同生成
│   └── pool.py              # 候选池管理
├── scripts/                 # 编排逻辑
│   ├── quality_gate.py      # 质量门 + 迭代循环
│   ├── strategy_state.py    # 策略状态机
│   └── style_learner.py     # 风格学习
├── feedback/                # 反馈收集
│   └── collector.py         # 反馈存储与查询
├── reporting/               # 输出渲染
│   └── renderer.py          # 文本/JSON/简报输出
├── tests/                   # 单元测试
├── examples/                # 示例 brief 文件
└── .state/                  # 运行时状态（自动生成）
    ├── strategy.json
    ├── style_profile.json
    └── feedback.json
```

## 依赖

- Python 3.10+
- PyYAML
- torch + transformers（仅当使用模型打分时需要）

## 运行方式

### CLI
```bash
python humanize.py examples/brief_wechat.json --verbose
python humanize.py examples/brief_email.json --format json --output-dir ./output
python humanize.py --status  # 查看系统状态
```

### 库调用
```python
from humanize import HumanizeEngine

engine = HumanizeEngine()
result = engine.run(spec, source_text)
print(result.final_text)
```

## 开发指南

- 添加替换规则：编辑 `heuristics/rules/*.yaml`
- 调整评分权重：修改 `scoring/scorer.py` 中的 `SCENARIO_WEIGHTS`
- 新增模板短语：编辑 `heuristics/rules/template_phrases.yaml`
- 添加新场景：在 `scorer.py` 的 `detect_scenario()` 和 `SCENARIO_WEIGHTS` 中添加

## 测试

```bash
python -m pytest tests/ -v
```
