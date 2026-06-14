"""FAISS 会话级目录隔离测试（不调用 Embedding API）。"""

import os


def test_faiss_dir_per_conversation(temp_storage):
    dir_a = temp_storage.get_faiss_dir_for_conversation(101)
    dir_b = temp_storage.get_faiss_dir_for_conversation(202)

    assert dir_a != dir_b
    assert dir_a.endswith("conv_101")
    assert dir_b.endswith("conv_202")


def test_delete_faiss_dir_for_conversation(temp_storage):
    cid = 999
    index_dir = temp_storage.get_faiss_dir_for_conversation(cid)
    os.makedirs(index_dir, exist_ok=True)

    marker = os.path.join(index_dir, "marker.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("test")

    assert os.path.isdir(index_dir)

    temp_storage.delete_faiss_dir_for_conversation(cid)
    assert not os.path.exists(index_dir)
