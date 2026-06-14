"""快速检查各 API Key 是否可用（不打印完整 Key）。"""
from dotenv import load_dotenv
import json
import os
import urllib.error
import urllib.request

load_dotenv()


def _mask(v: str) -> str:
    v = (v or "").strip().strip('"').strip("'")
    if not v:
        return "（未配置）"
    return f"{v[:6]}...{v[-4:]}" if len(v) > 12 else "（过短）"


def _post(url: str, headers: dict, payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, "OK"
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:300]


def main() -> None:
    ds = os.getenv("DEEPSEEK_API_KEY", "")
    sf = os.getenv("SILICONFLOW_API_KEY", "")
    tv = os.getenv("TAVILY_API_KEY", "")
    zp = os.getenv("ZHIPU_API_KEY", "")
    ls = os.getenv("LANGCHAIN_API_KEY", "").strip().strip('"').strip("'")
    tracing = (os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()

    print("=== Key 配置摘要（已脱敏）===")
    print("DEEPSEEK_API_KEY:", _mask(ds))
    print("SILICONFLOW_API_KEY:", _mask(sf))
    print("TAVILY_API_KEY:", _mask(tv))
    print("ZHIPU_API_KEY:", _mask(zp))
    print("LANGCHAIN_API_KEY:", _mask(ls))
    print("LANGCHAIN_TRACING_V2:", tracing)
    print()

    code, msg = _post(
        "https://api.deepseek.com/chat/completions",
        {"Authorization": f"Bearer {ds.strip()}"},
        {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
    )
    print(f"DeepSeek: HTTP {code} -> {'可用' if code == 200 else msg}")

    code, msg = _post(
        "https://api.siliconflow.cn/v1/embeddings",
        {"Authorization": f"Bearer {sf.strip()}"},
        {"model": "BAAI/bge-m3", "input": "test", "encoding_format": "float"},
    )
    print(f"SiliconFlow: HTTP {code} -> {'可用' if code == 200 else msg}")

    if zp.strip():
        code, msg = _post(
            "https://open.bigmodel.cn/api/paas/v4/web_search",
            {"Authorization": f"Bearer {zp.strip()}"},
            {"search_engine": "search_pro", "search_query": "test", "count": 1},
        )
        print(f"Zhipu WebSearch: HTTP {code} -> {'可用' if code == 200 else msg}")
    else:
        print("Zhipu WebSearch: 未配置 ZHIPU_API_KEY")

    if ls:
        req = urllib.request.Request(
            "https://api.smith.langchain.com/info",
            headers={"x-api-key": ls},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f"LangSmith: HTTP {r.status} -> 可用")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            print(f"LangSmith: HTTP {e.code} -> {body}")
            print("  提示: 若不需要链路追踪，把 .env 里 LANGCHAIN_TRACING_V2 改成 false")
    else:
        print("LangSmith: 未配置 LANGCHAIN_API_KEY")


if __name__ == "__main__":
    main()
