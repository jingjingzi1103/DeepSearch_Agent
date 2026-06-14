"""Server-Sent Events 格式化工具。"""

import json
from typing import Any, Dict


def format_sse_event(event: str, data: Dict[str, Any]) -> str:
    """将事件名与 JSON 数据格式化为 SSE 帧。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
