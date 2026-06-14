"""聊天历史格式转换（纯函数，便于单元测试）。"""

from typing import Dict, List


def convert_messages_for_agent(
    messages: List[Dict[str, str]],
    max_messages: int = 10,
    *,
    strip_assistant_for_realtime: bool = False,
) -> List[Dict[str, str]]:
    """将 UI 消息列表转为 LangChain 历史格式，并做滑动窗口截断。

    strip_assistant_for_realtime: 实时类问题时不把旧 assistant 回答喂给 Agent，避免复读幻觉。
    """
    if strip_assistant_for_realtime:
        messages = [m for m in messages if m.get("role") != "assistant"]

    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    history: List[Dict[str, str]] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            mapped_role = "human"
        elif role == "assistant":
            mapped_role = "ai"
        else:
            mapped_role = role
        history.append({"role": mapped_role, "content": m["content"]})
    return history
