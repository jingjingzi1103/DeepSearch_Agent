"""Agent 工具调用链路的序列化/反序列化（用于 SQLite 持久化与前端回放）。"""

import json
from typing import Any, Dict, List, Union

StepLike = Union[Dict[str, Any], Any]


def _serialize_observation(observation: Any) -> Any:
    if isinstance(observation, list):
        items: List[Any] = []
        for item in observation:
            if hasattr(item, "page_content"):
                items.append(
                    {
                        "_type": "document",
                        "page_content": (getattr(item, "page_content", "") or "")[:2000],
                        "metadata": getattr(item, "metadata", {}) or {},
                    }
                )
            elif isinstance(item, dict):
                items.append(item)
            else:
                items.append({"_type": "raw", "content": str(item)[:2000]})
        return items
    if isinstance(observation, dict):
        return observation
    return {"_type": "raw", "content": str(observation)[:4000]}


def serialize_intermediate_steps(intermediate_steps: List[Any]) -> List[Dict[str, Any]]:
    """将 LangChain intermediate_steps 转为可 JSON 存储的结构。"""
    out: List[Dict[str, Any]] = []
    for action, observation in intermediate_steps or []:
        out.append(
            {
                "tool": getattr(action, "tool", "unknown_tool"),
                "tool_input": getattr(action, "tool_input", ""),
                "observation": _serialize_observation(observation),
            }
        )
    return out


def normalize_steps(steps: List[StepLike]) -> List[Dict[str, Any]]:
    """统一 live / 已序列化 steps 为 dict 列表。"""
    normalized: List[Dict[str, Any]] = []
    for item in steps or []:
        if isinstance(item, dict) and "tool" in item:
            normalized.append(item)
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            action, observation = item
            normalized.append(
                {
                    "tool": getattr(action, "tool", "unknown_tool"),
                    "tool_input": getattr(action, "tool_input", ""),
                    "observation": observation,
                }
            )
    return normalized


def parse_json_tool_items(observation: Any, items_key: str = "items") -> List[Any]:
    """解析工具返回的 JSON 字符串 / dict 中的 items 列表。"""
    if isinstance(observation, str):
        try:
            parsed = json.loads(observation)
            if isinstance(parsed, dict):
                return parsed.get(items_key) or []
        except Exception:
            return []
    if isinstance(observation, dict):
        return observation.get(items_key) or []
    if isinstance(observation, list):
        return observation
    return []


def extract_sources_from_steps(steps: List[StepLike]) -> List[Dict[str, str]]:
    """从工具步骤中提取可展示的来源链接。"""
    sources: List[Dict[str, str]] = []
    seen = set()

    for step in normalize_steps(steps):
        tool_name = step.get("tool", "")
        observation = step.get("observation")

        if tool_name == "local_knowledge_search":
            obs = step.get("observation")
            docs = obs if isinstance(obs, list) else parse_json_tool_items(obs, items_key="items")
            for i, doc in enumerate(docs or [], start=1):
                content = ""
                if isinstance(doc, dict):
                    content = doc.get("page_content") or ""
                elif hasattr(doc, "page_content"):
                    content = getattr(doc, "page_content", "") or ""
                title = f"本地文档片段 {i}"
                snippet = (content or "").strip().replace("\n", " ")[:80]
                key = f"local::{i}::{snippet}"
                if key not in seen:
                    seen.add(key)
                    sources.append(
                        {
                            "title": title,
                            "url": "",
                            "source_type": "local_rag",
                            "snippet": snippet,
                        }
                    )
            continue

        if tool_name == "weibo_hot_search":
            for item in parse_json_tool_items(observation):
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or ""
                title = item.get("title") or item.get("word") or url or "微博热搜"
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": title, "url": url, "source_type": "weibo"})
            continue

        if tool_name == "zhipu_web_search":
            for item in parse_json_tool_items(observation):
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or item.get("link") or ""
                title = item.get("title") or url or "智谱搜索结果"
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": title, "url": url, "source_type": "zhipu"})
            continue

        if tool_name not in {"tavily_search_results_json", "tavily_search"}:
            continue

        if isinstance(observation, list):
            for item in observation:
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or ""
                title = item.get("title") or url or "未提供标题"
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": title, "url": url, "source_type": "web"})

    return sources


def build_trace_metadata(intermediate_steps: List[Any]) -> Dict[str, Any]:
    """构建 assistant 消息 metadata（sources + steps）。"""
    serialized = serialize_intermediate_steps(intermediate_steps)
    return {
        "sources": extract_sources_from_steps(serialized),
        "steps": serialized,
    }


def dumps_trace_metadata(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False)


def loads_trace_metadata(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
