"""FastAPI SSE 聊天接口测试（mock 业务层，不调用真实 LLM）。"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.agent_service import ChatTurnResult


def _parse_sse_events(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    for block in blocks:
        event_name = "message"
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_line = line.split(":", 1)[1].strip()
        if data_line:
            events.append((event_name, json.loads(data_line)))
    return events


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def temp_db(temp_storage):
    temp_storage.init_db()
    cid = temp_storage.create_conversation("API 测试会话")
    return cid


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_health_deep_endpoint(client):
    with patch("api.main.probe_deepseek_key", return_value=None):
        resp = client.get("/health/deep")
    assert resp.status_code == 200
    assert resp.json()["deepseek_key_ok"] is True


def test_chat_stream_rejects_empty_message(client, temp_db):
    resp = client.post(
        "/v1/chat/stream",
        json={"conversation_id": temp_db, "message": "   "},
    )
    assert resp.status_code == 400


@patch("api.routes.chat.build_agent")
@patch("api.routes.chat.assemble_agent_tools")
@patch("api.routes.chat.load_vector_store_for_conversation")
@patch("api.routes.chat.run_chat_turn")
def test_chat_stream_sse_event_sequence(
    mock_run_turn,
    mock_load_vs,
    mock_assemble,
    mock_build_agent,
    client,
    temp_db,
):
    mock_load_vs.return_value = MagicMock(vector_store=None, error=None)
    mock_assemble.return_value = [MagicMock(name="tavily_search_results_json")]
    mock_build_agent.return_value = MagicMock()

    mock_run_turn.return_value = ChatTurnResult(
        full_answer="这是测试回答",
        trace_metadata={"sources": [{"title": "src", "url": "http://x"}], "steps": []},
        eval_result={
            "faithfulness": 4,
            "relevance": 4,
            "has_citation": True,
            "overall": 4,
            "reason": "ok",
            "judge_model": "deepseek-chat",
        },
        used_forced_pipeline=False,
    )

    def _fake_run(*, on_status=None, on_answer_chunk=None, **kwargs):
        if on_status:
            on_status("正在处理...")
        if on_answer_chunk:
            on_answer_chunk("这是测试回答")
        return mock_run_turn.return_value

    mock_run_turn.side_effect = _fake_run

    resp = client.post(
        "/v1/chat/stream",
        json={
            "conversation_id": temp_db,
            "message": "你好",
            "enable_judge": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    events = _parse_sse_events(resp.text)
    event_names = [name for name, _ in events]
    assert "status" in event_names
    assert "token" in event_names
    assert "trace" in event_names
    assert "judge" in event_names
    assert event_names[-1] == "done"

    done_payload = events[-1][1]
    assert done_payload["conversation_id"] == temp_db
    assert done_payload["full_answer"] == "这是测试回答"
