"""历史消息滑动窗口测试。"""

from core.history_utils import convert_messages_for_agent


def _make_messages(n: int):
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"msg-{i}"})
    return out


def test_convert_history_keeps_last_10():
    history = convert_messages_for_agent(_make_messages(25), max_messages=10)

    assert len(history) == 10
    assert history[0]["content"] == "msg-15"
    assert history[-1]["content"] == "msg-24"


def test_convert_history_maps_roles():
    history = convert_messages_for_agent(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )

    assert history[0]["role"] == "human"
    assert history[1]["role"] == "ai"
