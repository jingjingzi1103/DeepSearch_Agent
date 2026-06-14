"""SSE 客户端解析与健康检查测试。"""

from unittest.mock import MagicMock, patch

from client.sse_client import (
    check_api_health,
    iter_sse_events,
    parse_sse_block,
)


def test_parse_sse_block():
    raw = 'event: token\ndata: {"delta": "你", "text": "你好"}\n'
    parsed = parse_sse_block(raw.strip())
    assert parsed is not None
    name, data = parsed
    assert name == "token"
    assert data["text"] == "你好"


def test_iter_sse_events_multiple_blocks():
    raw = (
        'event: status\ndata: {"message": "处理中"}\n\n'
        'event: done\ndata: {"conversation_id": 1, "full_answer": "ok"}\n\n'
    )
    events = list(iter_sse_events(raw))
    assert len(events) == 2
    assert events[0][0] == "status"
    assert events[1][0] == "done"


@patch("client.sse_client.httpx.Client")
def test_check_api_health_ok(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "ok", "deepseek_key_ok": True}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    ok, msg = check_api_health("http://127.0.0.1:8000")
    assert ok is True
    assert "已连接" in msg


@patch("client.sse_client.httpx.Client")
def test_check_api_health_connect_error(mock_client_cls):
    import httpx

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.side_effect = httpx.ConnectError("refused")
    mock_client_cls.return_value = mock_client

    ok, msg = check_api_health("http://127.0.0.1:8000")
    assert ok is False
    assert "无法连接" in msg
