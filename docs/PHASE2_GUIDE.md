# 阶段 2：LLM-as-Judge 质量评测

阶段 1 的 7 个 pytest 通过后，本阶段给 Agent 回答加上**自动质量打分**，面试时可讲「评测闭环」。

## 做了什么

| 文件 | 作用 |
|------|------|
| `core/evaluator.py` | Judge 逻辑：提取工具上下文 → 调 DeepSeek 打分 → 解析 JSON |
| `core/storage.py` | 新增 `evaluations` 表 + `save_evaluation_to_db` |
| `advanced_agent.py` | 每次回答后可选触发 Judge，侧边栏开关 |
| `tests/test_evaluator.py` | 3 个离线单测（不调 API） |

## 评分维度

- **faithfulness（忠实度）**：回答是否被检索上下文支撑，有无胡编
- **relevance（相关性）**：是否切题
- **has_citation（有引用）**：是否体现来源/引用意识
- **overall（综合）**：1–5 分
- **reason**：一句话理由

## 你需要做的（按顺序）

### 1. 跑测试（应 10 passed）

```powershell
cd d:\DeepSearch_Agent
python -m pytest tests/ -v
```

### 2. 启动应用

```powershell
streamlit run advanced_agent.py
```

侧边栏找到 **「启用 Judge 自动打分」**，默认开启。

### 3. 手动冒烟（建议跑 eval/cases.json 里 2 条 P0）

**P0-联网：**

> 请联网查询今天微博热搜前三是什么，并简要说明每条热搜的背景。

期望：有 Tavily 来源链接；Judge 展开后 faithfulness / relevance ≥ 3。

**P0-RAG（需先上传 PDF/TXT）：**

> 请总结这份文档的核心内容，并用要点列出主要贡献。

期望：调用 `local_knowledge_search`；忠实度应高于纯联网题。

### 4. 查数据库（可选，面试演示用）

```powershell
python -c "import sqlite3; c=sqlite3.connect('chat_history.db'); print(c.execute('SELECT overall, faithfulness, reason FROM evaluations ORDER BY id DESC LIMIT 3').fetchall())"
```

## 面试怎么说

> 我们不仅有功能测试（pytest + eval cases），还有 **LLM-as-Judge 在线评测**：每次回答后把「问题 + 工具检索上下文 + 最终回答」送给 Judge 模型，输出结构化分数并落库，形成可回归的质量基线。Judge 与主链路共用 DeepSeek，成本低；后续可换专用 Judge 模型或加 Rerank。

## 常见问题

**Judge 多调一次 API，会不会慢？**  
会多 2–5 秒。侧边栏可关；生产环境可改成异步队列。

**503 / 繁忙？**  
和主 Agent 一样，高峰期可能限流；Judge 失败会显示 warning，不影响主回答。

**要不要新 API Key？**  
不需要，默认复用 `.env` 里的 `DEEPSEEK_API_KEY`。

## 下一阶段（Phase 3 预告）

- FastAPI 后端 + Streamlit 纯客户端
- 批量跑 `eval/cases.json` 并出报告
- GitHub Actions CI
