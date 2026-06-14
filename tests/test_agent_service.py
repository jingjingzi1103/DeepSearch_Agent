"""agent_service 纯逻辑测试（不依赖 Streamlit）。"""

from unittest.mock import MagicMock, patch

from core.agent_service import (
    assemble_agent_tools,
    prepare_chat_history,
    build_vector_store_from_file,
)


class _FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content

    def getvalue(self):
        return self._content


def test_build_vector_store_unsupported_format():
    upload = _FakeUpload("readme.docx", b"fake")
    result = build_vector_store_from_file(upload, conversation_id=1)
    assert result.vector_store is None
    assert "仅支持 PDF 和 TXT" in (result.error or "")


def test_prepare_chat_history_strips_last_user():
    messages = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "current question"},
    ]
    history = prepare_chat_history(
        messages, "current question", local_rag_enabled=False
    )
    assert len(history) == 1
    assert history[0]["role"] in ("assistant", "ai")


def test_prepare_chat_history_strips_assistant_for_rag():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    history = prepare_chat_history(
        messages,
        "请根据文档总结对比实验",
        local_rag_enabled=True,
    )
    assert len(history) == 1
    assert history[0]["role"] in ("user", "human")


@patch("core.agent_service.get_tavily_tool")
@patch("core.agent_service.get_enabled_realtime_tools", return_value=[])
def test_assemble_agent_tools_respects_switches(mock_realtime, mock_tavily):
    mock_tavily.return_value = MagicMock(name="tavily_search")
    local = MagicMock()
    local.name = "local_knowledge_search"

    tools = assemble_agent_tools(
        enable_weibo_hot=True,
        enable_zhipu_search=True,
        local_tool=local,
    )
    names = [getattr(t, "name", "") for t in tools]
    assert "local_knowledge_search" in names
    mock_tavily.assert_called_once()
