# Advanced DeepSearch Agent

一个面向生产场景的多工具 AI Agent 项目，支持：
- DeepSeek 大模型对话
- Tavily 联网搜索
- 本地文档 RAG（PDF/TXT -> 切片 -> Embedding -> FAISS 检索）
- SQLite 会话记忆
- Streamlit 可视化交互

本项目包含两个入口文件：
- `advanced_agent.py`：主推荐版本（功能更完整、会话管理更强）
- `app.py`：早期版本/演示版本

---

## 功能特性

- **多工具 Agent**
  - 支持 `create_tool_calling_agent + AgentExecutor`
  - 动态工具路由：联网搜索 + 本地知识检索

- **真 RAG 流水线**
  - 文档解析：`PyPDFLoader` / TXT 文本读取
  - 文本切片：`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
  - 向量化：`OpenAIEmbeddings`（硅基流动 BGE-M3）
  - 检索：`FAISS` + `as_retriever(k=3)` + `create_retriever_tool`

- **会话级向量库持久化**
  - 每个会话独立 FAISS 目录（按会话隔离）
  - 刷新页面后可自动加载本会话对应向量库
  - 删除会话时同步删除该会话向量索引

- **本地数据库记忆**
  - SQLite 持久化聊天消息
  - 支持新建会话、切换会话、重命名、删除会话、清空历史

- **生产可用性增强**
  - 历史消息滑动窗口（最近 10 条）防止上下文过长
  - 生成期间禁用关键交互，减少并发冲突
  - 对 503 Busy 场景增加自动重试与指数退避
  - 支持 LangSmith 开关检测（预留追踪接入）

---

## 项目结构

```text
DeepSearch_Agent/
├── advanced_agent.py          # 主应用（推荐）
├── app.py                     # 早期版本
├── chat_history.db            # SQLite 聊天记录（运行后生成）
├── faiss_index_store/         # FAISS 向量索引目录（运行后生成）
└── .env                       # 环境变量
```

---

## 技术栈

- Python
- Streamlit
- LangChain
- FAISS
- SQLite
- DeepSeek API
- Tavily Search API
- SiliconFlow Embedding API（BAAI/bge-m3）

---

## 快速开始

## 1) 克隆项目

```bash
git clone <your-repo-url>
cd DeepSearch_Agent
```

## 2) 安装依赖

> 项目暂未提供 `requirements.txt`，可先按下列方式安装核心依赖。

```bash
pip install streamlit python-dotenv langchain langchain-openai langchain-community faiss-cpu pypdf
```

## 3) 配置环境变量

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key

# 可选：LangSmith 追踪（预留）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```

## 4) 启动应用

```bash
streamlit run advanced_agent.py
```

---

## 使用说明

1. 打开页面后，在左侧上传 PDF/TXT 文档（可选）。
2. 上传成功后，系统会自动：
   - 解析文档
   - 切片
   - Embedding
   - 构建并保存会话级 FAISS 索引
3. 在聊天框提问：
   - 文档相关问题优先走本地 `local_knowledge_search`
   - 实时问题走 Tavily 联网搜索
4. 支持会话管理：
   - 新建会话 / 切换会话 / 重命名 / 删除单会话 / 清空全部历史

---

## 关键设计说明

- **为什么不是完全实时 token 流式？**
  - 当前使用 `AgentExecutor`，更偏“步骤级流式”。
  - 已通过前端渐进渲染优化观感，后续可升级为底层 LLM 真流式架构。

- **为什么会遇到 403/503？**
  - `503` 常见于服务端繁忙，已加入自动重试机制。
  - `403 RPM limit exceeded` 是配额/频率限制问题，需要控制请求频率或升级 API 配额（如完成实名认证）。

- **Embedding 413 报错说明**
  - 已对 embedding 批量大小做限制（`chunk_size=64`），避免单批输入过大导致 413。

---

## 常见问题（FAQ）

### Q1：刷新后文档为什么“看起来没加载”？
- 当前逻辑按“会话”加载向量库。
- 需确认当前会话是否存在对应 `faiss_index_store/conv_<id>/` 目录。

### Q2：切换会话后为什么回答风格变化？
- 每个会话的历史消息与文档索引独立，属于预期行为。

### Q3：能否支持更多文档格式？
- 可以，建议扩展为 DOCX/Markdown/网页 URL 的统一 Loader 层。

---

## 后续可迭代方向

- 真正 token 级流式输出（SSE/WebSocket）
- 检索质量增强（Reranker、MMR、元数据过滤）
- 多文档知识库管理（同会话多文档）
- 引用溯源增强（回答中标注来源段落/页码）
- 更完整的观测体系（LangSmith + 自定义日志指标）

---

## License

可根据你的开源计划补充（例如 MIT / Apache-2.0）。

