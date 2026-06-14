"""LLM-as-Judge：支持多 Provider 交叉评测（DeepSeek / 智谱）。"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from core.trace_utils import parse_json_tool_items

JUDGE_PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "zhipu": {
        "label": "智谱 GLM",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4-flash",
    },
}


def resolve_judge_config(provider: Optional[str] = None) -> Dict[str, str]:
    """解析 Judge Provider 与模型名。"""
    name = (provider or os.getenv("JUDGE_PROVIDER") or "deepseek").strip().lower()
    if name not in JUDGE_PROVIDERS:
        raise ValueError(
            f"不支持的 JUDGE_PROVIDER: {name}，可选: {', '.join(JUDGE_PROVIDERS)}"
        )
    spec = JUDGE_PROVIDERS[name]
    model = (os.getenv("JUDGE_MODEL") or spec["default_model"]).strip()
    # 若显式指定了 provider，且 JUDGE_MODEL 仍是另一家的默认名，则用当前 provider 默认模型
    if provider and not os.getenv("JUDGE_MODEL"):
        model = spec["default_model"]
    return {
        "provider": name,
        "label": spec["label"],
        "model": model,
        "display": f"{spec['label']} / {model}",
    }


def list_available_judge_providers() -> List[Dict[str, str]]:
    """返回已配置 API Key、可用的 Judge Provider 列表。"""
    from core.env_loader import load_project_env

    load_project_env()
    available: List[Dict[str, str]] = []
    for name, spec in JUDGE_PROVIDERS.items():
        if (os.getenv(spec["api_key_env"]) or "").strip():
            cfg = resolve_judge_config(name)
            available.append(
                {
                    "provider": name,
                    "label": spec["label"],
                    "model": cfg["model"],
                    "display": cfg["display"],
                }
            )
    return available


def get_judge_llm(provider: Optional[str] = None):
    """按 Provider 创建 Judge LLM 客户端。"""
    from langchain_openai import ChatOpenAI

    from core.env_loader import get_env_key

    cfg = resolve_judge_config(provider)
    spec = JUDGE_PROVIDERS[cfg["provider"]]
    api_key = get_env_key(spec["api_key_env"])

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=spec["base_url"],
        model=cfg["model"],
        temperature=0.0,
    )


def extract_context_from_steps(intermediate_steps: List[Any]) -> str:
    """从 Agent 中间步骤提取可供 Judge 参考的上下文摘要。"""
    blocks: List[str] = []
    for idx, (action, observation) in enumerate(intermediate_steps or [], start=1):
        tool_name = getattr(action, "tool", "unknown")
        blocks.append(f"[工具 {idx}] {tool_name}")

        if tool_name == "local_knowledge_search" and isinstance(observation, list):
            for i, doc in enumerate(observation[:3], start=1):
                content = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else ""
                if isinstance(doc, dict):
                    content = doc.get("page_content", "")
                snippet = (content or "").strip().replace("\n", " ")
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                blocks.append(f"  文档片段{i}: {snippet}")
        elif tool_name == "weibo_hot_search":
            for item in parse_json_tool_items(observation)[:5]:
                if isinstance(item, dict):
                    blocks.append(
                        f"  热搜{item.get('rank','?')}: {item.get('title','')} "
                        f"(热度 {item.get('heat','')}, {item.get('fetched_at','')})"
                    )
        elif tool_name == "zhipu_web_search":
            for item in parse_json_tool_items(observation)[:5]:
                if isinstance(item, dict):
                    blocks.append(
                        f"  [{item.get('rank','?')}] {item.get('title','')} | "
                        f"{item.get('url','')} | {str(item.get('content',''))[:120]}"
                    )
        elif isinstance(observation, list):
            for item in observation[:3]:
                if isinstance(item, dict):
                    title = item.get("title") or "未提供标题"
                    url = item.get("url") or ""
                    snippet = item.get("content") or item.get("snippet") or ""
                    snippet = str(snippet).replace("\n", " ")[:200]
                    blocks.append(f"  网页: {title} | {url} | {snippet}")
        else:
            blocks.append(f"  返回节选: {str(observation)[:400]}")

    return "\n".join(blocks) if blocks else "（本次未调用工具或未返回检索上下文）"


def _parse_judge_json(text: str, *, judge_model: str, judge_provider: str) -> Dict[str, Any]:
    """从 Judge 输出中解析 JSON（兼容 markdown 代码块包裹）。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    data = json.loads(text)
    return {
        "faithfulness": float(data.get("faithfulness", 0)),
        "relevance": float(data.get("relevance", 0)),
        "has_citation": bool(data.get("has_citation", False)),
        "overall": float(data.get("overall", 0)),
        "reason": str(data.get("reason", "")),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
    }


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    *,
    judge_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """调用 Judge 模型对回答打分。

    返回字段：faithfulness, relevance, has_citation, overall, reason,
              judge_provider, judge_model
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    cfg = resolve_judge_config(judge_provider)
    llm = get_judge_llm(judge_provider)

    system = (
        "你是一名严格的中文 AI 回答质量评测员。"
        "请仅根据提供的「问题、检索上下文、最终回答」进行客观评分。"
        "如果上下文不足以支撑回答，faithfulness 应偏低。"
        "必须只输出 JSON，不要输出其他文字。格式如下：\n"
        '{"faithfulness":1-5,"relevance":1-5,"has_citation":true/false,'
        '"overall":1-5,"reason":"一句话理由"}'
    )
    user = (
        f"【用户问题】\n{question}\n\n"
        f"【检索/工具上下文】\n{context}\n\n"
        f"【最终回答】\n{answer}\n\n"
        "请评分。"
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        return _parse_judge_json(
            content,
            judge_model=cfg["display"],
            judge_provider=cfg["provider"],
        )
    except Exception as e:
        return {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "has_citation": False,
            "overall": 0.0,
            "reason": f"Judge 评测失败: {e}",
            "judge_provider": cfg["provider"],
            "judge_model": cfg["display"],
            "error": True,
        }
