"""RAG 强制检索测试。"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from core.intent_utils import is_document_rag_query
from core.rag_pipeline import run_rag_pipeline


class _FakeDoc:
    def __init__(self, content: str):
        self.page_content = content
        self.metadata = {"source": "test.pdf", "page": 1}


@pytest.fixture()
def _fake_langchain_messages(monkeypatch):
    lc_messages = types.SimpleNamespace(
        SystemMessage=lambda content: ("system", content),
        HumanMessage=lambda content: ("human", content),
    )
    monkeypatch.setitem(sys.modules, "langchain_core.messages", lc_messages)
    monkeypatch.setitem(sys.modules, "langchain_core", types.SimpleNamespace(messages=lc_messages))


def test_is_document_rag_query():
    assert is_document_rag_query("请总结这篇论文的对比实验", local_rag_enabled=True)
    assert is_document_rag_query("4. 对比的意义", local_rag_enabled=True)
    assert not is_document_rag_query("今天微博热搜前三", local_rag_enabled=True)
    assert not is_document_rag_query("总结论文", local_rag_enabled=False)


def test_run_rag_pipeline_with_mock_store(_fake_langchain_messages):
    doc = _FakeDoc("Lag-Llama beats baseline in comparison experiments.")
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    store = MagicMock()
    store.as_retriever.return_value = retriever

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="According to snippet 1, the comparison shows...")

    answer, steps, meta = run_rag_pipeline("What does the comparison experiment show?", llm, store, k=3)
    assert "comparison" in answer.lower() or "snippet" in answer.lower()
    assert len(steps) == 1
    assert steps[0][0].tool == "local_knowledge_search"
    assert meta["steps"][0]["tool"] == "local_knowledge_search"
    assert any(s.get("source_type") == "local_rag" for s in meta.get("sources", []))
