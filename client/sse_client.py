"""消费 FastAPI SSE 流式对话接口。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, Optional, Tuple

import httpx

from client.http_client import api_client

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def get_api_base_url() -> str:
    return (os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL).rstrip("/")


def parse_sse_block(block: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """解析单个 SSE 事件块（event + data）。"""
    event_name = "message"
    data_line = ""
    for line in block.splitlines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_line = line.split(":", 1)[1].strip()
    if not data_line:
        return None
    return event_name, json.loads(data_line)


def iter_sse_events(raw_text: str) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """从完整 SSE 文本中迭代事件（用于测试）。"""
    for block in [b.strip() for b in raw_text.split("\n\n") if b.strip()]:
        parsed = parse_sse_block(block)
        if parsed:
            yield parsed


def check_api_health(base_url: Optional[str] = None, timeout: float = 10.0) -> Tuple[bool, str]:
    """检查 API 是否可用（轻量 /health，不探测 DeepSeek）。"""
    url = (base_url or get_api_base_url()).rstrip("/")
    try:
        with api_client(timeout) as client:
            resp = client.get(f"{url}/health")
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") not in {"ok", "degraded"}:
                return False, f"API 状态异常：{body}"
            return True, "API 已连接"
    except httpx.ConnectError:
        return False, (
            f"无法连接 API（{url}）。请先运行：\n"
            "uvicorn api.main:app --reload --port 8000 --reload-dir api --reload-dir core"
        )
    except httpx.ReadTimeout:
        return False, (
            f"API 响应超时（{url}）。若刚改代码触发了 reload，请等几秒后点 Streamlit 右上角 Rerun。"
        )
    except Exception as e:
        return False, f"API 健康检查失败：{e}"


def stream_chat(
    *,
    conversation_id: int,
    message: str,
    enable_weibo_hot: bool = True,
    enable_zhipu_search: bool = True,
    enable_forced_rag: bool = True,
    enable_judge: bool = True,
    judge_provider: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 600.0,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """调用 POST /v1/chat/stream，逐条 yield (event_name, data)。"""
    url = (base_url or get_api_base_url()).rstrip("/")
    payload = {
        "conversation_id": conversation_id,
        "message": message,
        "enable_weibo_hot": enable_weibo_hot,
        "enable_zhipu_search": enable_zhipu_search,
        "enable_forced_rag": enable_forced_rag,
        "enable_judge": enable_judge,
        "judge_provider": judge_provider,
    }

    with api_client(timeout) as client:
        with client.stream("POST", f"{url}/v1/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            event_name: Optional[str] = None
            data_parts: list[str] = []

            for line in resp.iter_lines():
                if line is None:
                    continue
                if line == "":
                    if data_parts:
                        yield (event_name or "message", json.loads("".join(data_parts)))
                    event_name = None
                    data_parts = []
                    continue

                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_parts.append(line.split(":", 1)[1].strip())
