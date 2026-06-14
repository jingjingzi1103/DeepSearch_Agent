"""RAG 强制检索链路：已上传文档时先检索向量库，再基于片段生成回答。"""

from typing import Any, Dict, List, Tuple

from core.trace_utils import build_trace_metadata


class _SyntheticAction:
    def __init__(self, tool: str, tool_input: Any):
        self.tool = tool
        self.tool_input = tool_input


def _format_docs_for_prompt(docs: List[Any]) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs or [], start=1):
        content = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else ""
        if isinstance(doc, dict):
            content = doc.get("page_content", "")
        meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        if isinstance(doc, dict):
            meta = doc.get("metadata", {}) or {}
        snippet = (content or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        blocks.append(f"[片段 {i}] metadata={meta}\n{snippet}")
    return "\n\n".join(blocks) if blocks else "（未检索到任何文档片段）"


def run_rag_pipeline(
    user_input: str,
    llm: Any,
    vector_store: Any,
    *,
    k: int = 5,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    """强制调用本地 FAISS 检索，再让 LLM 仅基于检索片段组织回答。"""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(user_input)
    if docs is None:
        docs = []

    action = _SyntheticAction("local_knowledge_search", user_input)
    steps: List[Any] = [(action, docs)]
    trace_metadata = build_trace_metadata(steps)
    chunks_text = _format_docs_for_prompt(docs)

    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "你是文档问答助手。你必须且只能根据【local_knowledge_search 返回的文档片段】回答，"
        "禁止使用模型预训练记忆或对话历史中的未验证信息。"
        "若片段不足以回答，请明确说明「文档片段中未找到」，不得编造。"
        "回答中可用 [片段1][片段2] 标注依据。"
    )
    user = (
        f"【用户问题】\n{user_input}\n\n"
        f"【local_knowledge_search 检索片段】\n{chunks_text}\n\n"
        "请基于以上片段作答。"
    )
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return answer, steps, trace_metadata
