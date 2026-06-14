"""API 请求/响应模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    conversation_id: int = Field(..., ge=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户提问")
    enable_weibo_hot: bool = True
    enable_zhipu_search: bool = True
    enable_forced_rag: bool = True
    enable_judge: bool = True
    judge_provider: Optional[str] = Field(
        None, description="Judge Provider：deepseek 或 zhipu，默认读 JUDGE_PROVIDER"
    )


class HealthResponse(BaseModel):
    status: str
    deepseek_key_ok: bool
    deepseek_key_error: Optional[str] = None


class ChatDonePayload(BaseModel):
    conversation_id: int
    full_answer: str
    used_forced_pipeline: bool = False
    error: Optional[str] = None


class ChatStreamEventTypes:
    STATUS = "status"
    TOKEN = "token"
    TRACE = "trace"
    JUDGE = "judge"
    DONE = "done"
    ERROR = "error"


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1)


class ConversationItem(BaseModel):
    id: int
    title: str
    preview: str = ""


class MessageItem(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class EvaluationItem(BaseModel):
    question: str
    answer: str
    faithfulness: float = 0
    relevance: float = 0
    has_citation: bool = False
    overall: float = 0
    reason: str = ""
    judge_model: str = ""
    created_at: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    conversation_id: int
    has_index: bool
    index_dir: Optional[str] = None


class JudgeProviderItem(BaseModel):
    provider: str
    label: str
    model: str
    display: str


class DocumentUploadResponse(BaseModel):
    conversation_id: int
    filename: str
    success: bool
    error: Optional[str] = None
    warning: Optional[str] = None
