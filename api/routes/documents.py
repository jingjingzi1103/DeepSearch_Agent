"""文档上传与 RAG 向量库 API。"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import DocumentStatusResponse, DocumentUploadResponse
from core.agent_service import build_vector_store_from_file
from core.storage import get_faiss_dir_for_conversation, list_conversations_with_preview

router = APIRouter(prefix="/conversations", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".txt"}


class _BytesUpload:
    """适配 build_vector_store_from_file 的文件对象。"""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content

    def getvalue(self):
        return self._content


def _ensure_conversation_exists(conversation_id: int) -> None:
    convs = {c["id"] for c in list_conversations_with_preview()}
    if int(conversation_id) not in convs:
        raise HTTPException(status_code=404, detail=f"会话 {conversation_id} 不存在")


@router.get("/{conversation_id}/documents/status", response_model=DocumentStatusResponse)
def document_status(conversation_id: int) -> DocumentStatusResponse:
    _ensure_conversation_exists(conversation_id)
    index_dir = get_faiss_dir_for_conversation(conversation_id)
    exists = os.path.isdir(index_dir)
    return DocumentStatusResponse(
        conversation_id=conversation_id,
        has_index=exists,
        index_dir=index_dir if exists else None,
    )


@router.post("/{conversation_id}/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    conversation_id: int,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    _ensure_conversation_exists(conversation_id)

    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = os.path.splitext(filename.lower())[1]
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="目前仅支持 PDF 和 TXT 文件。")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    adapter = _BytesUpload(filename, content)
    result = build_vector_store_from_file(adapter, conversation_id=conversation_id)

    if result.warning and result.vector_store is None:
        return DocumentUploadResponse(
            conversation_id=conversation_id,
            filename=filename,
            success=False,
            warning=result.warning,
            error=result.error,
        )

    if result.vector_store is None:
        return DocumentUploadResponse(
            conversation_id=conversation_id,
            filename=filename,
            success=False,
            error=result.error or "构建向量库失败",
            warning=result.warning,
        )

    return DocumentUploadResponse(
        conversation_id=conversation_id,
        filename=filename,
        success=True,
        warning=result.warning,
    )
