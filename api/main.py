"""FastAPI 应用入口。

启动方式：
    uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router
from api.routes.conversations import router as conversations_router
from api.routes.documents import router as documents_router
from api.routes.judge import router as judge_router
from api.schemas import HealthResponse
from core.agent_service import init_env, probe_deepseek_key
from core.storage import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_env()
    init_db()
    yield


app = FastAPI(
    title="DeepSearch Agent API",
    description="Phase C 后端：SSE 流式对话",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/v1")
app.include_router(conversations_router, prefix="/v1")
app.include_router(documents_router, prefix="/v1")
app.include_router(judge_router, prefix="/v1")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """轻量存活探针：不调用外部 API，避免阻塞 worker 导致 502。"""
    return HealthResponse(
        status="ok",
        deepseek_key_ok=True,
        deepseek_key_error=None,
    )


@app.get("/health/deep", response_model=HealthResponse)
def health_deep() -> HealthResponse:
    """深度健康检查：探测 DeepSeek Key（较慢，仅供运维/调试）。"""
    key_err = probe_deepseek_key()
    return HealthResponse(
        status="ok" if key_err is None else "degraded",
        deepseek_key_ok=key_err is None,
        deepseek_key_error=key_err,
    )
