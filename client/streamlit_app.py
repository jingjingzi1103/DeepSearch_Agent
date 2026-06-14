"""薄 Streamlit 客户端：全部业务经 FastAPI（对话 SSE + 会话/文档 REST）。

启动前请先运行后端：
    uvicorn api.main:app --reload --port 8000 --reload-dir api --reload-dir core

启动客户端（在项目根目录执行）：
    streamlit run client/streamlit_app.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from client import api_client
from client.components import (
    init_session_state,
    render_chat_history,
    render_eval_panel,
    render_quick_start_buttons,
    render_sidebar,
    render_trace_panel,
)
from client.sse_client import stream_chat


def main() -> None:
    st.set_page_config(
        page_title="DeepSearch Agent (SSE Client)",
        page_icon="🔍",
        layout="wide",
    )

    init_session_state()

    api_ok = render_sidebar()
    if not api_ok:
        st.warning("请先启动 FastAPI 后端，再刷新本页面。")
        st.code(
            "uvicorn api.main:app --reload --port 8000 --reload-dir api --reload-dir core",
            language="bash",
        )
        st.stop()

    st.title("🔍 DeepSearch Agent")
    st.caption("前后端分离 · 会话/文档/对话均由 API 处理")

    render_chat_history()
    render_quick_start_buttons()

    pending = (st.session_state.get("pending_user_input") or "").strip()
    user_input = st.chat_input(
        "请输入你的问题…",
        disabled=bool(st.session_state.get("is_generating", False)),
    )
    if not user_input and pending:
        user_input = pending
        st.session_state["pending_user_input"] = ""

    if not user_input:
        return

    cid = int(st.session_state["conversation_id"])

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()
        trace_slot = st.empty()
        judge_slot = st.empty()

        st.session_state["is_generating"] = True
        full_answer = ""
        trace_metadata: dict = {}
        eval_result = None
        error = None
        preflight_shown = False

        try:
            for event_name, data in stream_chat(
                conversation_id=cid,
                message=user_input,
                enable_weibo_hot=st.session_state.get("enable_weibo_hot", True),
                enable_zhipu_search=st.session_state.get("enable_zhipu_search", True),
                enable_forced_rag=st.session_state.get("enable_forced_rag", True),
                enable_judge=st.session_state.get("enable_judge", True),
                judge_provider=st.session_state.get("judge_provider"),
            ):
                if event_name == "status":
                    msg = data.get("message", "")
                    kind = data.get("kind", "")
                    if kind == "preflight" and not preflight_shown:
                        status_placeholder.info(msg)
                        preflight_shown = True
                    elif kind != "preflight":
                        status_placeholder.info(msg)
                elif event_name == "token":
                    full_answer = data.get("text") or full_answer
                    if full_answer:
                        answer_placeholder.markdown(full_answer)
                elif event_name == "trace":
                    trace_metadata = {
                        "sources": data.get("sources") or [],
                        "steps": data.get("steps") or [],
                    }
                elif event_name == "judge":
                    eval_result = data
                elif event_name == "error":
                    error = data.get("message") or "未知错误"
                elif event_name == "done":
                    full_answer = data.get("full_answer") or full_answer
                    if data.get("error"):
                        error = data["error"]

        except Exception as e:
            error = f"调用 API 失败：{e}"
        finally:
            st.session_state["is_generating"] = False

        status_placeholder.empty()

        if error:
            st.error(error)
            return

        if full_answer:
            answer_placeholder.markdown(full_answer)

        sources = trace_metadata.get("sources") or []
        steps = trace_metadata.get("steps") or []
        if sources or steps:
            with trace_slot.container():
                render_trace_panel(sources, steps)

        if eval_result:
            with judge_slot.container():
                render_eval_panel(eval_result)

    st.session_state["messages"] = api_client.get_messages(cid)
    st.rerun()


if __name__ == "__main__":
    main()
