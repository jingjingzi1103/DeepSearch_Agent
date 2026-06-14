"""聊天 SSE 路由。"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain.agents import AgentExecutor

from api.schemas import ChatStreamEventTypes, ChatStreamRequest
from api.sse import format_sse_event
from core.agent_service import (
    ChatTurnResult,
    assemble_agent_tools,
    build_agent,
    build_retriever_tool,
    load_vector_store_for_conversation,
    run_chat_turn,
)
from core.platform_guard import get_preflight_notice
from core.storage import (
    load_history_from_db,
    save_evaluation_to_db,
    save_message_to_db,
)

router = APIRouter(tags=["chat"])

QueueItem = Union[str, tuple[str, ChatTurnResult], None]


@dataclass
class ChatRuntimeContext:
    vector_store: Any
    local_tool: Any
    local_rag_enabled: bool
    agent_executor: AgentExecutor
    enabled_tool_names: List[str]


def _build_runtime_context(request: ChatStreamRequest) -> ChatRuntimeContext:
    load_result = load_vector_store_for_conversation(request.conversation_id)
    vector_store = load_result.vector_store
    local_tool = build_retriever_tool(vector_store) if vector_store is not None else None
    local_rag_enabled = vector_store is not None and local_tool is not None

    tools = assemble_agent_tools(
        enable_weibo_hot=request.enable_weibo_hot,
        enable_zhipu_search=request.enable_zhipu_search,
        local_tool=local_tool,
    )
    agent_executor = build_agent(tools)
    enabled_tool_names = [getattr(t, "name", "") for t in tools]

    return ChatRuntimeContext(
        vector_store=vector_store,
        local_tool=local_tool,
        local_rag_enabled=local_rag_enabled,
        agent_executor=agent_executor,
        enabled_tool_names=enabled_tool_names,
    )


def _run_chat_in_thread(
    *,
    request: ChatStreamRequest,
    messages: List[Dict[str, Any]],
    runtime: ChatRuntimeContext,
    event_q: queue.Queue[QueueItem],
) -> None:
    """在后台线程执行 run_chat_turn，通过队列推送 SSE 帧。"""
    last_answer_len = 0

    def on_status(msg: str) -> None:
        event_q.put(format_sse_event(ChatStreamEventTypes.STATUS, {"message": msg}))

    def on_answer_chunk(text: str) -> None:
        nonlocal last_answer_len
        delta = text[last_answer_len:]
        last_answer_len = len(text)
        if delta:
            event_q.put(
                format_sse_event(
                    ChatStreamEventTypes.TOKEN,
                    {"delta": delta, "text": text},
                )
            )

    try:
        turn = run_chat_turn(
            user_input=request.message,
            agent_executor=runtime.agent_executor,
            messages=messages,
            vector_store=runtime.vector_store,
            local_rag_enabled=runtime.local_rag_enabled,
            enable_weibo_hot=request.enable_weibo_hot,
            enable_forced_rag=request.enable_forced_rag,
            enable_judge=request.enable_judge,
            judge_provider=request.judge_provider,
            enabled_tool_names=runtime.enabled_tool_names,
            on_status=on_status,
            on_answer_chunk=on_answer_chunk,
        )
        event_q.put(("__result__", turn))
    except Exception as e:
        event_q.put(("__result__", ChatTurnResult(error=str(e))))
    finally:
        event_q.put(None)


def generate_chat_sse(request: ChatStreamRequest) -> Generator[str, None, None]:
    """生成一轮对话的 SSE 事件流。"""
    runtime = _build_runtime_context(request)

    preflight = get_preflight_notice(request.message, runtime.enabled_tool_names)
    if preflight:
        yield format_sse_event(
            ChatStreamEventTypes.STATUS,
            {"message": preflight, "kind": "preflight"},
        )

    save_message_to_db(request.conversation_id, "user", request.message)
    messages = load_history_from_db(request.conversation_id)

    event_q: queue.Queue[QueueItem] = queue.Queue()
    worker = threading.Thread(
        target=_run_chat_in_thread,
        kwargs={
            "request": request,
            "messages": messages,
            "runtime": runtime,
            "event_q": event_q,
        },
        daemon=True,
    )
    worker.start()

    turn: Optional[ChatTurnResult] = None

    while True:
        item = event_q.get()
        if item is None:
            break
        if isinstance(item, tuple) and item[0] == "__result__":
            turn = item[1]
            continue
        yield item

    worker.join(timeout=0.1)

    if turn is None:
        turn = ChatTurnResult(error="对话执行未返回结果")

    if turn.error:
        yield format_sse_event(ChatStreamEventTypes.ERROR, {"message": turn.error})
        yield format_sse_event(
            ChatStreamEventTypes.DONE,
            {
                "conversation_id": request.conversation_id,
                "full_answer": "",
                "error": turn.error,
            },
        )
        return

    trace_metadata = turn.trace_metadata or {}
    if trace_metadata.get("sources") or trace_metadata.get("steps"):
        yield format_sse_event(
            ChatStreamEventTypes.TRACE,
            {
                "sources": trace_metadata.get("sources") or [],
                "steps": trace_metadata.get("steps") or [],
            },
        )

    if turn.eval_result:
        save_evaluation_to_db(
            request.conversation_id,
            request.message,
            turn.full_answer,
            turn.eval_result,
        )
        yield format_sse_event(ChatStreamEventTypes.JUDGE, turn.eval_result)

    save_message_to_db(
        request.conversation_id,
        "assistant",
        turn.full_answer,
        trace_metadata=trace_metadata,
    )

    yield format_sse_event(
        ChatStreamEventTypes.DONE,
        {
            "conversation_id": request.conversation_id,
            "full_answer": turn.full_answer,
            "used_forced_pipeline": turn.used_forced_pipeline,
        },
    )


@router.post("/chat/stream")
def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
    """流式对话：SSE 事件类型 status / token / trace / judge / done / error。"""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    return StreamingResponse(
        generate_chat_sse(request.model_copy(update={"message": message})),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
