"""微博实时热搜 Provider（无需 API Key，走公开 ajax 接口）。"""

import datetime
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


WEIBO_HOT_URL = "https://weibo.com/ajax/side/hotSearch"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://weibo.com/",
}


def fetch_weibo_hot_search(limit: int = 10) -> List[Dict[str, Any]]:
    """拉取微博实时热搜榜。"""
    limit = max(1, min(int(limit), 50))
    req = urllib.request.Request(WEIBO_HOT_URL, headers=DEFAULT_HEADERS, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"微博热搜接口 HTTP {e.code}") from e
    except Exception as e:
        raise RuntimeError(f"微博热搜接口请求失败: {e}") from e

    realtime = (payload.get("data") or {}).get("realtime") or []
    fetched_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(realtime[:limit], start=1):
        if not isinstance(row, dict):
            continue
        word = (row.get("word") or row.get("note") or "").strip()
        if not word:
            continue
        heat = row.get("num") or row.get("raw_hot") or ""
        tag = row.get("icon_desc") or row.get("label_name") or ""
        q = urllib.parse.quote(word)
        items.append(
            {
                "rank": idx,
                "title": word,
                "word": word,
                "heat": heat,
                "tag": tag,
                "url": f"https://s.weibo.com/weibo?q={q}",
                "source": "weibo.com",
                "fetched_at": fetched_at,
            }
        )

    if not items:
        raise RuntimeError("微博热搜接口返回为空")
    return items


class WeiboHotInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=50, description="返回热搜条数，默认 10")


def _weibo_hot_search(limit: int = 10) -> str:
    items = fetch_weibo_hot_search(limit=limit)
    return json.dumps(
        {"provider": "weibo_hot_search", "items": items, "count": len(items)},
        ensure_ascii=False,
    )


def build_weibo_hot_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_weibo_hot_search,
        name="weibo_hot_search",
        description=(
            "获取微博实时热搜榜（官方 ajax 接口）。"
            "当用户询问微博热搜、今日热搜前三、微博热点排行时必须优先调用此工具，"
            "不要凭记忆或旧新闻回答。"
        ),
        args_schema=WeiboHotInput,
    )
