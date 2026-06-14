# Phase C：FastAPI + SSE 分离（计划）

冗余清理已完成（删除 `app.py`、移除 dead code）。下一步按此顺序推进：

## Step 1 — 抽取 `core/agent_service.py` ✅ 已完成

- 从 `advanced_agent.py` 迁出：Agent 构建、工具组装、强制微博/RAG 链路、Judge
- `advanced_agent.py` 只保留 Streamlit UI（调用 `run_chat_turn`）
- 新增 `tests/test_agent_service.py`，pytest 31 项全绿

## Step 2 — FastAPI 骨架 + SSE ✅ 已完成

- `api/main.py`、`api/routes/chat.py`、`api/schemas.py`、`api/sse.py`
- `GET /health` — 健康检查（含 DeepSeek Key 探测）
- `POST /v1/chat/stream` → SSE 事件：`status` / `token` / `trace` / `judge` / `done` / `error`
- 启动：`uvicorn api.main:app --reload --port 8000`

## Step 3 — 薄 Streamlit 客户端 ✅ 已完成

- `client/streamlit_app.py` — 新推荐入口，消费 SSE 逐 token 渲染
- `client/sse_client.py` — httpx SSE 客户端 + API 健康检查
- `client/components.py` — UI 组件（侧边栏、历史、trace、Judge）
- `advanced_agent.py` — 兼容旧入口，转发到 `client/streamlit_app.py`
- 文档上传仍走本地（Step 4 再迁 API）；对话与存库由 API 负责

## Step 4 — 会话与上传 API ✅ 已完成

- `GET/POST/DELETE /v1/conversations`
- `GET /v1/conversations/{id}/messages`、`PATCH` 重命名、`DELETE` 删除
- `GET /v1/conversations/{id}/evaluations`
- `POST /v1/conversations/{id}/documents/upload`、`GET .../documents/status`
- `client/api_client.py` — Streamlit 经 REST 管理会话与上传，不再直连 SQLite/FAISS

## SSE vs WebSocket

- 首版用 **SSE**（实现简单、够用于流式回答）
- 后续如需「停止生成」再补 WebSocket
