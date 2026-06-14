"""文档上传 API 测试。"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from core.agent_service import VectorStoreBuildResult


def test_document_status_no_index(temp_storage):
    temp_storage.init_db()
    cid = temp_storage.create_conversation("doc test")
    client = TestClient(app)

    resp = client.get(f"/v1/conversations/{cid}/documents/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_index"] is False


@patch("api.routes.documents.build_vector_store_from_file")
def test_document_upload_success(mock_build, temp_storage):
    temp_storage.init_db()
    cid = temp_storage.create_conversation("upload test")
    mock_build.return_value = VectorStoreBuildResult(vector_store=MagicMock())

    client = TestClient(app)
    resp = client.post(
        f"/v1/conversations/{cid}/documents/upload",
        files={"file": ("notes.txt", b"hello rag content", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    mock_build.assert_called_once()
