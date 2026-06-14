"""用户意图识别（实时数据 / 微博热搜等）。"""

import re
from typing import Optional

_WEIBO_HOT_KEYWORDS = (
    "微博热搜",
    "微博热榜",
    "微博热点",
    "热搜榜",
    "热搜前三",
    "热搜前3",
    "热搜前十",
    "热搜前10",
    "热搜排行",
    "weibo热搜",
    "weibo hot",
)


def is_weibo_hot_query(text: str) -> bool:
    """是否为微博热搜类实时查询。"""
    t = (text or "").strip().lower()
    if not t:
        return False
    if "微博" in t and any(k in t for k in ("热搜", "热榜", "热点", "排行")):
        return True
    return any(k.lower() in t for k in _WEIBO_HOT_KEYWORDS)


def is_realtime_query(text: str) -> bool:
    """广义实时信息查询（需强制走工具，不可复用历史回答）。"""
    t = (text or "").strip()
    if is_weibo_hot_query(t):
        return True
    markers = ("今天", "此刻", "刚刚", "实时", "最新", "当前", "现在")
    hot_markers = ("热搜", "热榜", "热点新闻", "热点")
    return any(m in t for m in markers) and any(h in t for h in hot_markers)


def extract_weibo_limit(text: str, default: int = 10) -> int:
    """从用户问题中解析需要的热搜条数。"""
    t = text or ""
    m = re.search(r"前\s*(\d+)", t)
    if m:
        return max(1, min(int(m.group(1)), 50))
    if "前三" in t or "前3" in t or "top3" in t.lower():
        return 3
    if "前五" in t or "前5" in t:
        return 5
    if "前十" in t or "前10" in t:
        return 10
    return default


_DOCUMENT_KEYWORDS = (
    "文档",
    "论文",
    "这篇",
    "这份",
    "该文",
    "本文",
    "pdf",
    "根据文档",
    "上传",
    "核心内容",
    "摘要",
    "结论",
    "方法",
    "实验",
    "对比",
    "图表",
    "表格",
    "章节",
    "附录",
    "贡献",
    "作者",
    "baseline",
    "消融",
    "数据集",
    "指标",
    "性能",
    "模型",
    "讲了什么",
    "主要内容",
    "总结",
    "要点",
    "概述",
)


def is_document_rag_query(text: str, *, local_rag_enabled: bool = False) -> bool:
    """是否为应强制走本地文档检索的问题（需已上传并向量化）。"""
    if not local_rag_enabled:
        return False
    t = (text or "").strip()
    if not t:
        return False
    if is_weibo_hot_query(t):
        return False
    web_only_markers = ("微博热搜", "抖音热", "热榜前三", "联网查", "热点新闻", "今天新闻")
    if any(m in t for m in web_only_markers) and not any(k in t for k in ("文档", "论文")):
        return False
    if any(k in t for k in _DOCUMENT_KEYWORDS):
        return True
    if re.search(r"^\d+[\.、]\s*", t):
        return True
    if any(k in t for k in ("上述", "前面提到", "这一节", "该节", "本节")):
        return True
    return False
