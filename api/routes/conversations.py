"""会话管理 REST API。"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ConversationCreateRequest,
    ConversationItem,
    ConversationRenameRequest,
    EvaluationItem,
    MessageItem,
)
from core.storage import (
    clear_all_history,
    create_conversation,
    delete_conversation,
    load_evaluations_for_conversation,
    load_history_from_db,
    list_conversations_with_preview,
    save_message_to_db,
    update_conversation_title,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

WELCOME_MESSAGE = (
    "你好，我是你的多引擎 AI Agent（前后端分离版）。\n"
    "对话由 FastAPI 后端处理；左侧可上传 PDF/TXT 启用本地 RAG。"
)


def _ensure_conversation_exists(conversation_id: int) -> None:
    convs = {c["id"] for c in list_conversations_with_preview()}
    if int(conversation_id) not in convs:
        raise HTTPException(status_code=404, detail=f"会话 {conversation_id} 不存在")


@router.get("", response_model=List[ConversationItem])
def list_conversations() -> List[ConversationItem]:
    return [ConversationItem(**c) for c in list_conversations_with_preview()]


@router.post("", response_model=ConversationItem)
def create_new_conversation(body: ConversationCreateRequest) -> ConversationItem:
    title = (body.title or "").strip()
    if not title:
        now = datetime.datetime.now()
        title = f"新会话 {now:%Y-%m-%d %H:%M}"
    cid = create_conversation(title)
    save_message_to_db(cid, "assistant", WELCOME_MESSAGE)
    return ConversationItem(id=cid, title=title, preview=WELCOME_MESSAGE[:30] + "...")


@router.delete("")
def clear_conversations() -> Dict[str, str]:
    clear_all_history()
    return {"status": "cleared"}


@router.get("/{conversation_id}/messages", response_model=List[MessageItem])
def get_messages(conversation_id: int) -> List[MessageItem]:
    _ensure_conversation_exists(conversation_id)
    rows = load_history_from_db(conversation_id)
    return [
        MessageItem(
            role=m["role"],
            content=m["content"],
            metadata=m.get("metadata"),
        )
        for m in rows
    ]


@router.get("/{conversation_id}/evaluations", response_model=List[EvaluationItem])
def get_evaluations(conversation_id: int) -> List[EvaluationItem]:
    _ensure_conversation_exists(conversation_id)
    return [EvaluationItem(**e) for e in load_evaluations_for_conversation(conversation_id)]


@router.patch("/{conversation_id}", response_model=ConversationItem)
def rename_conversation(
    conversation_id: int, body: ConversationRenameRequest
) -> ConversationItem:
    _ensure_conversation_exists(conversation_id)
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title 不能为空")
    update_conversation_title(conversation_id, title)
    conv = next(
        (c for c in list_conversations_with_preview() if int(c["id"]) == int(conversation_id)),
        None,
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return ConversationItem(**conv)


@router.delete("/{conversation_id}")
def remove_conversation(conversation_id: int) -> Dict[str, Any]:
    _ensure_conversation_exists(conversation_id)
    delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}
