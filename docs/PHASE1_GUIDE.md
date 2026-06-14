# 阶段 1：质量保障入门（一步步操作指南）

## 你需要知道的第一件事

**阶段 1 不需要申请新的 API Key。**

本阶段目标是：建立 **评测用例 + 自动化测试（pytest）**，不调用 DeepSeek / Tavily / Embedding。

---

## 第 0 步：确认项目能跑起来

```powershell
cd d:\DeepSearch_Agent
pip install -r requirements.txt
streamlit run advanced_agent.py
```

浏览器打开 http://localhost:8501 ，能聊天即可。

---

## 第 1 步：安装 pytest

```powershell
pip install pytest
```

---

## 第 2 步：运行自动化测试（阶段 1 核心）

```powershell
cd d:\DeepSearch_Agent
pytest tests/ -v
```

你应该看到 **7 passed**（或类似全部通过）。

这些测试在验证什么？

| 测试文件 | 验证内容 |
|----------|----------|
| `test_conversation_storage.py` | SQLite 会话消息读写、会话隔离、删除 |
| `test_faiss_paths.py` | 每个会话独立 FAISS 目录、删除索引 |
| `test_history_window.py` | 历史消息只保留最近 10 条 |

**面试话术：**

> “我把会话存储、向量库路径、历史窗口做成了可自动化回归的单测，不依赖外部 API，CI 里可以稳定跑。”

---

## 第 3 步：看懂评测用例清单

打开 `eval/cases.json`，这是 **手工冒烟 + 未来自动化 E2E** 的用例清单。

每条包含：
- `id`：用例编号
- `category`：smoke / consistency / rag_quality / stability
- `priority`：P0 / P1
- `input`：测试输入
- `expect_tool`：期望调用的工具

你可以慢慢从 8 条扩到 30～50 条。

---

## 第 4 步：手工冒烟（5 分钟）

按 `eval/cases.json` 里 P0 用例，在网页上点一遍：

1. 基础问答能返回  
2. 上传文档后能文档问答  
3. 实时问题能联网  
4. 切换会话不串消息/文档  
5. 刷新后历史还在  

---

## 第 5 步：项目结构变化（面试可讲）

我们新增了：

```text
core/
  storage.py        # 数据库 + FAISS 路径（无 LangChain，可单测）
  history_utils.py  # 历史消息滑动窗口（纯函数）
eval/
  cases.json        # 评测/冒烟用例清单
tests/
  conftest.py
  test_*.py
docs/
  PHASE1_GUIDE.md   # 本文件
```

这叫 **“把可测逻辑从 UI 里拆出来”**，是工程化第一步。

---

## 阶段 2 预告：Judge 打分（那时才考虑新 API）

阶段 1 完成后，再做 **LLM-as-Judge**。三种选择：

| 方案 | 是否需要新 Key | 说明 |
|------|----------------|------|
| A | 否 | 仍用 DeepSeek，换 Judge 专用 Prompt |
| B | 可选 | 用你已有的 **硅基流动** 再调一个模型做 Judge |
| C | 是 | 新申请：通义 / 智谱 / OpenAI 等，专门做交叉评测 |

**推荐顺序：** 先做 A（零成本验证流程），稳定后再做 B 或 C。

常见申请地址（自行注册）：
- 硅基流动：https://siliconflow.cn（你已有 Key）
- 智谱：https://open.bigmodel.cn
- 通义：https://dashscope.aliyun.com

---

## 你下一步做什么？

1. 本地跑通：`pytest tests/ -v` 全部绿色  
2. 手工跑一遍 P0 冒烟  
3. 回复我 **「阶段 1 跑通了」**，我带你做 **阶段 2：evaluator.py + Judge 打分**
