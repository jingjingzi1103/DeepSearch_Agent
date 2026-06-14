"""实时数据强制链路：不依赖 LLM 是否自觉调工具，先拉 API 再生成回答。"""

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from core.realtime.weibo import fetch_weibo_hot_search
from core.trace_utils import build_trace_metadata


class _SyntheticAction:
    def __init__(self, tool: str, tool_input: Any):
        self.tool = tool
        self.tool_input = tool_input


def _build_weibo_steps(limit: int, payload_json: str) -> List[Any]:
    action = _SyntheticAction("weibo_hot_search", {"limit": limit})
    return [(action, payload_json)]


def run_weibo_hot_pipeline(user_input: str, llm: Any, limit: Optional[int] = None) -> Tuple[str, List[Any], Dict[str, Any]]:
    """强制调用微博热搜 API，再让 LLM 仅基于工具结果组织回答。"""
    from core.intent_utils import extract_weibo_limit

    limit = limit or extract_weibo_limit(user_input, default=10)
    items = fetch_weibo_hot_search(limit=limit)
    payload = {
        "provider": "weibo_hot_search",
        "items": items,
        "count": len(items),
        "note": "数据来自 weibo.com/ajax/side/hotSearch 实时接口",
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    steps = _build_weibo_steps(limit, payload_json)
    trace_metadata = build_trace_metadata(steps)

    system = (
        "你是数据整理助手。你必须且只能根据【工具返回的 JSON】回答，禁止使用对话记忆或旧新闻。"
        "按排名列出热搜词条、热度、标签、抓取时间 fetched_at；可简要概括背景但不得编造事实。"
        "若 JSON 中 items 为空，如实说明接口无数据。"
    )
    user = (
        f"【用户问题】\n{user_input}\n\n"
        f"【weibo_hot_search 工具返回】\n{payload_json}\n\n"
        "请基于以上 JSON 作答。"
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer, steps, trace_metadata


def agent_used_tool(intermediate_steps: List[Any], tool_name: str) -> bool:
    for action, _ in intermediate_steps or []:
        if getattr(action, "tool", "") == tool_name:
            return True
    return False
