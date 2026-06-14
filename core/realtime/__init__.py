"""实时数据 Provider 注册中心（可扩展接入更多第三方 API）。"""

import os
from typing import Any, List

from core.env_loader import get_env_key


def _is_enabled(env_name: str, default: str = "true") -> bool:
    flag = (os.getenv(env_name) or default).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def get_enabled_realtime_tools() -> List[Any]:
    """按 .env 开关返回已启用的实时/联网数据工具。"""
    tools: List[Any] = []

    if _is_enabled("ENABLE_WEIBO_HOT"):
        from core.realtime.weibo import build_weibo_hot_tool

        tools.append(build_weibo_hot_tool())

    if _is_enabled("ENABLE_ZHIPU_SEARCH", "true"):
        try:
            get_env_key("ZHIPU_API_KEY")
            from core.realtime.zhipu_search import build_zhipu_web_search_tool

            tools.append(build_zhipu_web_search_tool())
        except ValueError:
            pass

    return tools
