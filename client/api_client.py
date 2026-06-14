"""REST API 客户端：会话管理与文档上传。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx

from client.http_client import api_client
from client.sse_client import get_api_base_url


class ApiClientError(Exception):
    pass


def _url(path: str, base_url: Optional[str] = None) -> str:
    return f"{(base_url or get_api_base_url()).rstrip('/')}{path}"


def _handle_response(resp: httpx.Response) -> Any:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text[:200]
        raise ApiClientError(f"HTTP {resp.status_code}: {detail}") from e
    if resp.status_code == 204 or not resp.content:
        return None
    return resp.json()


def list_conversations(base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    with api_client() as client:
        return _handle_response(client.get(_url("/v1/conversations", base_url)))


def create_conversation(title: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
    with api_client() as client:
        return _handle_response(
            client.post(_url("/v1/conversations", base_url), json={"title": title})
        )


def clear_all_conversations(base_url: Optional[str] = None) -> None:
    with api_client() as client:
        _handle_response(client.delete(_url("/v1/conversations", base_url)))


def get_messages(conversation_id: int, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    with api_client() as client:
        return _handle_response(
            client.get(_url(f"/v1/conversations/{conversation_id}/messages", base_url))
        )


def get_evaluations(conversation_id: int, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    with api_client() as client:
        return _handle_response(
            client.get(_url(f"/v1/conversations/{conversation_id}/evaluations", base_url))
        )


def rename_conversation(
    conversation_id: int, title: str, base_url: Optional[str] = None
) -> Dict[str, Any]:
    with api_client() as client:
        return _handle_response(
            client.patch(
                _url(f"/v1/conversations/{conversation_id}", base_url),
                json={"title": title},
            )
        )


def delete_conversation(conversation_id: int, base_url: Optional[str] = None) -> None:
    with api_client() as client:
        _handle_response(client.delete(_url(f"/v1/conversations/{conversation_id}", base_url)))


def get_document_status(conversation_id: int, base_url: Optional[str] = None) -> Dict[str, Any]:
    with api_client() as client:
        return _handle_response(
            client.get(_url(f"/v1/conversations/{conversation_id}/documents/status", base_url))
        )


def upload_document(
    conversation_id: int,
    filename: str,
    content: bytes,
    base_url: Optional[str] = None,
    timeout: float = 600.0,
) -> Tuple[bool, Dict[str, Any]]:
    with api_client(timeout) as client:
        resp = client.post(
            _url(f"/v1/conversations/{conversation_id}/documents/upload", base_url),
            files={"file": (filename, content)},
        )
        data = _handle_response(resp)
        return bool(data.get("success")), data
