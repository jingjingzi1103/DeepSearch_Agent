"""智谱 Web Search Provider（需 ZHIPU_API_KEY）。"""

import datetime
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.env_loader import get_env_key

ZHIPU_WEB_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
DEFAULT_SEARCH_ENGINE = os.getenv("ZHIPU_SEARCH_ENGINE", "search_pro")


def fetch_zhipu_web_search(
    query: str,
    *,
    count: int = 8,
    search_engine: Optional[str] = None,
    search_recency_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """调用智谱 Web Search API，返回标准化结果列表。"""
    query = (query or "").strip()
    if not query:
        raise ValueError("搜索词不能为空")

    count = max(1, min(int(count), 50))
    api_key = get_env_key("ZHIPU_API_KEY")

    body: Dict[str, Any] = {
        "search_engine": search_engine or DEFAULT_SEARCH_ENGINE,
        "search_query": query,
        "count": count,
    }
    if search_recency_filter:
        body["search_recency_filter"] = search_recency_filter

    req = urllib.request.Request(
        ZHIPU_WEB_SEARCH_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"智谱 Web Search HTTP {e.code}: {detail}") from e
    except Exception as e:
        raise RuntimeError(f"智谱 Web Search 请求失败: {e}") from e

    if payload.get("error"):
        raise RuntimeError(f"智谱 Web Search 错误: {payload.get('error')}")

    raw_items = payload.get("search_result") or []
    fetched_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items: List[Dict[str, Any]] = []

    for idx, row in enumerate(raw_items, start=1):
        if not isinstance(row, dict):
            continue
        title = (row.get("title") or "").strip()
        url = (row.get("link") or row.get("url") or "").strip()
        if not title and not url:
            continue
        content = (row.get("content") or "")[:800]
        items.append(
            {
                "rank": idx,
                "title": title or url,
                "url": url,
                "content": content,
                "publish_date": row.get("publish_date") or "",
                "refer": row.get("refer") or "",
                "source": "zhipu_web_search",
                "fetched_at": fetched_at,
            }
        )

    if not items:
        raise RuntimeError("智谱 Web Search 未返回有效结果")
    return items


class ZhipuSearchInput(BaseModel):
    query: str = Field(description="搜索关键词或完整问句，建议不超过 70 字")
    count: int = Field(default=8, ge=1, le=20, description="返回条数，默认 8")


def _zhipu_web_search(query: str, count: int = 8) -> str:
    items = fetch_zhipu_web_search(query, count=count)
    return json.dumps(
        {
            "provider": "zhipu_web_search",
            "query": query,
            "items": items,
            "count": len(items),
        },
        ensure_ascii=False,
    )


def build_zhipu_web_search_tool():
    from langchain.tools import StructuredTool

    return StructuredTool.from_function(
        func=_zhipu_web_search,
        name="zhipu_web_search",
        description=(
            "智谱 AI 联网搜索（Web Search API）。"
            "适用于中文新闻、国内热点、行业动态、人物/公司产品等需要联网检索的问题。"
            "当问题不是微博热搜榜时，可优先于 Tavily 使用本工具。"
        ),
        args_schema=ZhipuSearchInput,
    )
