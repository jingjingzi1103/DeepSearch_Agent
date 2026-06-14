"""平台能力边界检测：避免用 A 平台搜索结果推断 B 平台热榜。"""

from typing import Any, Dict, List, Optional

from core.intent_utils import is_weibo_hot_query
from core.trace_utils import normalize_steps, parse_json_tool_items


def is_douyin_hot_query(text: str) -> bool:
    """用户是否在问抖音热榜 / 是否上热门。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    has_douyin = "抖音" in t or "douyin" in t
    has_hot = any(k in t for k in ("热榜", "热搜", "上榜", "上热", "热门", "火了", "流量"))
    return has_douyin and has_hot


def is_person_activity_query(text: str) -> bool:
    """是否在问某人近况/动态（可能跨平台）。"""
    t = text or ""
    markers = ("知不知道", "是谁", "最近", "动向", "动态", "发了", "有没有")
    return any(m in t for m in markers)


def _collect_trace_blob(steps: List[Any]) -> str:
    """把工具返回合并为可检索文本（小写）。"""
    parts: List[str] = []
    for step in normalize_steps(steps):
        tool = step.get("tool", "")
        parts.append(str(tool))
        parts.append(str(step.get("tool_input", "")))
        obs = step.get("observation")
        if isinstance(obs, str):
            parts.append(obs)
        elif isinstance(obs, (dict, list)):
            parts.append(str(obs))
        if tool == "zhipu_web_search":
            for item in parse_json_tool_items(obs):
                if isinstance(item, dict):
                    parts.extend(
                        [
                            str(item.get("title", "")),
                            str(item.get("url", "")),
                            str(item.get("content", "")),
                            str(item.get("refer", "")),
                        ]
                    )
        if tool == "weibo_hot_search":
            for item in parse_json_tool_items(obs):
                if isinstance(item, dict):
                    parts.extend([str(item.get("title", "")), str(item.get("word", ""))])
    return "\n".join(parts).lower()


def _trace_used_tool(steps: List[Any], tool_name: str) -> bool:
    return any(step.get("tool") == tool_name for step in normalize_steps(steps))


def _has_platform_evidence(trace_blob: str, platform: str) -> bool:
    keywords = {
        "douyin": ("抖音", "douyin", "tiktok"),
        "weibo": ("微博", "weibo", "s.weibo.com"),
    }
    keys = keywords.get(platform, ())
    return any(k.lower() in trace_blob for k in keys)


def _has_hot_list_evidence(trace_blob: str) -> bool:
    return any(k in trace_blob for k in ("热榜", "热搜", "hot", "trending", "上榜"))


def build_platform_disclaimer(user_input: str, trace_metadata: Optional[Dict[str, Any]]) -> str:
    """若问题涉及某平台热榜但检索无直接依据，返回需附加的说明（空串表示无需）。"""
    steps = (trace_metadata or {}).get("steps") or []
    if not steps:
        return ""

    blob = _collect_trace_blob(steps)
    notes: List[str] = []

    if is_douyin_hot_query(user_input):
        if not _trace_used_tool(steps, "douyin_hot_search"):
            notes.append(
                "当前系统**未接入抖音实时热榜 API**，仅有智谱/Tavily 等通用网页搜索。"
            )
        if not (_has_platform_evidence(blob, "douyin") and _has_hot_list_evidence(blob)):
            notes.append(
                "检索结果中**未找到可核实「抖音热榜/是否上热门」的直接依据**，"
                "因此无法确认该人物/内容是否登上抖音热榜。"
            )
        if notes:
            notes.append(
                "若下文提到其他平台（如微博、新闻）的动态，均来自检索到的网页来源，"
                "**不能**据此推断抖音热榜状态。"
            )
            return "⚠️ **数据边界说明**\n\n" + "\n".join(f"- {n}" for n in notes)

    if is_weibo_hot_query(user_input) and not _trace_used_tool(steps, "weibo_hot_search"):
        if not (_has_platform_evidence(blob, "weibo") and _has_hot_list_evidence(blob)):
            notes.append(
                "未调用微博实时热搜接口，且检索结果中缺乏可核实的微博热榜依据，"
                "关于微博热搜排名的结论请谨慎参考。"
            )
            return "⚠️ **数据边界说明**\n\n" + "\n".join(f"- {n}" for n in notes)

    return ""


def get_preflight_notice(user_input: str, enabled_tools: List[str]) -> Optional[str]:
    """提问开始时给用户的能力边界提示。"""
    if is_douyin_hot_query(user_input) and "douyin_hot_search" not in enabled_tools:
        return (
            "您的问题涉及 **抖音热榜**。当前仅支持 **微博实时热搜** + **智谱网页搜索**，"
            "无法直接查询抖音榜单；我会基于网页检索介绍可查到的公开报道，"
            "并在无法证实时明确说明。"
        )
    return None
