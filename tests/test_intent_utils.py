"""意图识别与历史过滤测试。"""

from core.history_utils import convert_messages_for_agent
from core.intent_utils import extract_weibo_limit, is_document_rag_query, is_realtime_query, is_weibo_hot_query


def test_is_weibo_hot_query():
    assert is_weibo_hot_query("今天微博热搜前三是什么")
    assert is_weibo_hot_query("微博热榜")
    assert not is_weibo_hot_query("介绍一下 DeepSeek")


def test_extract_weibo_limit():
    assert extract_weibo_limit("热搜前三") == 3
    assert extract_weibo_limit("前5") == 5
    assert extract_weibo_limit("前20") == 20


def test_strip_assistant_for_realtime_history():
    messages = [
        {"role": "user", "content": "昨天的问题"},
        {"role": "assistant", "content": "旧榜单 A B C"},
        {"role": "user", "content": "今天微博热搜前三"},
    ]
    history = convert_messages_for_agent(messages, strip_assistant_for_realtime=True)
    roles = [h["role"] for h in history]
    assert "ai" not in roles
    assert history[-1]["content"] == "今天微博热搜前三"


def test_is_realtime_query():
    assert is_realtime_query("今天的热点新闻")
    assert not is_realtime_query("介绍 RAG 原理")


def test_is_document_rag_query_followup():
    assert is_document_rag_query("4. 对比的意义", local_rag_enabled=True)
    assert not is_document_rag_query("4. 对比的意义", local_rag_enabled=False)
