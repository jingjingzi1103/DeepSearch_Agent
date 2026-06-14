"""会话与 SQLite 持久化相关测试（不调用外部 AI API）。"""


def test_create_conversation_and_save_message(temp_storage):
    temp_storage.init_db()

    cid = temp_storage.create_conversation("测试会话")
    temp_storage.save_message_to_db(cid, "user", "你好")
    temp_storage.save_message_to_db(
        cid,
        "assistant",
        "你好，我是 Agent",
        trace_metadata={"sources": [{"title": "src", "url": "https://a.com"}], "steps": []},
    )

    history = temp_storage.load_history_from_db(cid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"].startswith("你好，我是")
    assert history[1]["metadata"]["sources"][0]["url"] == "https://a.com"


def test_sessions_are_isolated_in_db(temp_storage):
    temp_storage.init_db()

    cid_a = temp_storage.create_conversation("会话A")
    cid_b = temp_storage.create_conversation("会话B")

    temp_storage.save_message_to_db(cid_a, "user", "只属于A")
    temp_storage.save_message_to_db(cid_b, "user", "只属于B")

    history_a = temp_storage.load_history_from_db(cid_a)
    history_b = temp_storage.load_history_from_db(cid_b)

    assert len(history_a) == 1
    assert history_a[0]["content"] == "只属于A"
    assert history_b[0]["content"] == "只属于B"


def test_delete_conversation_removes_messages(temp_storage):
    temp_storage.init_db()

    cid = temp_storage.create_conversation("待删除会话")
    temp_storage.save_message_to_db(cid, "user", "临时消息")

    temp_storage.delete_conversation(cid)
    history = temp_storage.load_history_from_db(cid)

    assert history == []
