"""本地 API 的 httpx 客户端（绕过 Windows 系统代理）。"""

import httpx


def api_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, trust_env=False)
