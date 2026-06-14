"""Judge 评分持久化测试。"""


def test_save_and_load_evaluations(temp_storage):
    temp_storage.init_db()

    cid = temp_storage.create_conversation("评测会话")
    temp_storage.save_message_to_db(cid, "user", "今天热搜？")
    temp_storage.save_message_to_db(cid, "assistant", "热搜 A、B、C")

    temp_storage.save_evaluation_to_db(
        cid,
        "今天热搜？",
        "热搜 A、B、C",
        {
            "faithfulness": 3.0,
            "relevance": 4.0,
            "has_citation": True,
            "overall": 3.5,
            "reason": "来源时效性不足",
            "judge_model": "deepseek-chat",
        },
    )

    rows = temp_storage.load_evaluations_for_conversation(cid)
    assert len(rows) == 1
    assert rows[0]["overall"] == 3.5
    assert rows[0]["question"] == "今天热搜？"
    assert rows[0]["answer"] == "热搜 A、B、C"


def test_delete_conversation_removes_evaluations(temp_storage):
    temp_storage.init_db()

    cid = temp_storage.create_conversation("待删")
    temp_storage.save_evaluation_to_db(
        cid,
        "q",
        "a",
        {"faithfulness": 5, "relevance": 5, "has_citation": False, "overall": 5, "reason": "ok"},
    )

    temp_storage.delete_conversation(cid)
    assert temp_storage.load_evaluations_for_conversation(cid) == []
