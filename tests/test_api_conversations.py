"""会话 REST API 测试。"""

from fastapi.testclient import TestClient

from api.main import app


def test_conversation_crud_flow(temp_storage):
    temp_storage.init_db()
    client = TestClient(app)

    listed = client.get("/v1/conversations")
    assert listed.status_code == 200

    created = client.post("/v1/conversations", json={"title": "测试会话"})
    assert created.status_code == 200
    cid = created.json()["id"]

    msgs = client.get(f"/v1/conversations/{cid}/messages")
    assert msgs.status_code == 200
    assert len(msgs.json()) >= 1

    renamed = client.patch(f"/v1/conversations/{cid}", json={"title": "新标题"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "新标题"

    deleted = client.delete(f"/v1/conversations/{cid}")
    assert deleted.status_code == 200
