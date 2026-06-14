"""pytest 公共配置：使用临时目录，避免污染真实 chat_history.db / faiss_index_store。"""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def temp_storage(monkeypatch):
    """为每个测试用例提供独立的 DB 与 FAISS 根目录。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "test_chat_history.db"
        faiss_root = tmp_path / "faiss_index_store"
        faiss_root.mkdir(parents=True, exist_ok=True)

        import core.storage as storage

        monkeypatch.setattr(storage, "DB_PATH", str(db_path))
        monkeypatch.setattr(storage, "FAISS_ROOT_DIR", str(faiss_root))

        yield storage
