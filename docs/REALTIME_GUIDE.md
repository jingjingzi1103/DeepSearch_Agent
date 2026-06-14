# 实时数据专项 + Trace 持久化

## 新增能力

| 模块 | 作用 |
|------|------|
| `core/realtime/weibo.py` | 微博实时热搜（公开 ajax，**无需 Key**） |
| `core/realtime/zhipu_search.py` | 智谱 Web Search（需 **ZHIPU_API_KEY**） |
| `core/realtime/__init__.py` | 实时工具注册，后续可接更多 API |
| `core/trace_utils.py` | 工具调用链序列化 + 来源提取 |
| `messages.trace_metadata` | SQLite 持久化来源 & 思考过程 |

## 前端展示

每条 **assistant** 回答下方（刷新后仍在）：

1. **🧾 来源链接 & 思考过程** — Tavily / 微博热搜 / RAG 片段
2. **📊 回答质量评分** — Judge 分数（上一阶段已有）

侧边栏：

- **启用微博实时热搜** — 默认开；关则 fallback Tavily

## .env 配置

```env
ENABLE_WEIBO_HOT=true
ZHIPU_API_KEY=your_zhipu_api_key
ENABLE_ZHIPU_SEARCH=true
```

## 验证智谱搜索

侧边栏打开 **「启用智谱联网搜索」**，提问：

> 最近一周 AI 领域有什么热点新闻？

展开 **🧾 来源链接 & 思考过程**，应看到 `zhipu_web_search` 步骤。

```powershell
python scripts/check_keys.py   # 应显示 Zhipu WebSearch: HTTP 200
```

## 三个 Provider 怎么分工

| 工具 | 场景 | Key |
|------|------|-----|
| `weibo_hot_search` | **微博**热搜榜（强制 Pipeline） | 不需要 |
| `zhipu_web_search` | 中文新闻、人物报道、**非榜单**类检索 | ZHIPU_API_KEY |
| `tavily_search` | 英文/通用联网 fallback | TAVILY_API_KEY |

> **抖音 / 小红书 / B站热榜**：当前**未接入**实时榜单 API。问「某人抖音是否上热榜」时，智谱只能搜到**全网新闻报道**，无法核实抖音榜单；系统会自动附加「数据边界说明」，避免跨平台瞎推断。

## 接入下一个 API（模板）

1. 在 `core/realtime/` 新建 `your_api.py`
2. 实现 `fetch_xxx()` + `build_xxx_tool()`（LangChain `StructuredTool`）
3. 在 `core/realtime/__init__.py` 的 `get_enabled_realtime_tools()` 里注册
4. 在 `core/trace_utils.py` 的 `extract_sources_from_steps` 增加来源解析
5. 在 `advanced_agent.py` 的 `render_tool_steps_content` 增加展示分支

若新 API 需要 Key，在 `.env` 增加 `YOUR_API_KEY=`，用 `core/env_loader.get_env_key()` 读取。

## 面试怎么说

> 实时榜单类需求不能靠通用搜索，我们做了 **Realtime Provider 插件层**：微博走官方 ajax 实时榜，Tavily 做泛搜索 fallback；工具调用链和来源 **落库 + 前端回放**，Judge 可基于真实 trace 打分，形成「检索—回答—评测—可追溯」闭环。

## 验证

```powershell
streamlit run advanced_agent.py
```

1. 点 **今天微博热搜前三** → 应调用 `weibo_hot_search`，展开思考过程可见排名
2. **刷新页面** → 来源 & 思考过程 & Judge 分数仍在
3. `python -m pytest tests/ -v` → 应全部通过
