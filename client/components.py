"""Streamlit UI 组件（纯 API 驱动，不直连 SQLite/FAISS）。"""

import datetime
from typing import Any, Dict, List

import streamlit as st

from client import api_client
from client.sse_client import check_api_health, get_api_base_url
from core.trace_utils import normalize_steps, parse_json_tool_items


def init_session_state() -> None:
    """通过 API 初始化会话状态（API 不可用时仅设置默认值）。"""
    st.session_state.setdefault("is_generating", False)
    st.session_state.setdefault("pending_user_input", "")
    st.session_state.setdefault("rag_upload_fingerprint", None)
    st.session_state.setdefault("rag_has_index", False)
    st.session_state.setdefault("messages", [])

    ok, _ = check_api_health()
    if not ok:
        st.session_state.setdefault("conversation_id", 1)
        return

    try:
        conversations = api_client.list_conversations()
    except Exception:
        st.session_state.setdefault("conversation_id", 1)
        return

    if "conversation_id" not in st.session_state:
        if conversations:
            st.session_state["conversation_id"] = conversations[0]["id"]
        else:
            created = api_client.create_conversation()
            st.session_state["conversation_id"] = created["id"]

    if not st.session_state.get("messages"):
        cid = int(st.session_state["conversation_id"])
        st.session_state["messages"] = api_client.get_messages(cid)

    _refresh_rag_status()


def _refresh_rag_status() -> None:
    cid = int(st.session_state["conversation_id"])
    try:
        status = api_client.get_document_status(cid)
        st.session_state["rag_has_index"] = bool(status.get("has_index"))
    except Exception:
        st.session_state["rag_has_index"] = False


def render_eval_panel(eval_result: Dict[str, Any]) -> None:
    with st.expander("📊 回答质量评分（LLM-as-Judge）", expanded=False):
        cols = st.columns(4)
        cols[0].metric("忠实度", f"{eval_result.get('faithfulness', 0):.1f}/5")
        cols[1].metric("相关性", f"{eval_result.get('relevance', 0):.1f}/5")
        cols[2].metric("综合", f"{eval_result.get('overall', 0):.1f}/5")
        cols[3].metric("有引用", "是" if eval_result.get("has_citation") else "否")
        judge_label = eval_result.get("judge_model") or eval_result.get("judge_provider") or "未知"
        st.caption(f"Judge 模型：{judge_label}")
        if eval_result.get("reason"):
            st.caption(f"理由：{eval_result['reason']}")
        if eval_result.get("error"):
            st.warning("Judge 调用失败，分数仅供参考。")


def render_trace_panel(sources: List[Dict[str, str]], steps: List[Any]) -> None:
    if not sources and not steps:
        return
    with st.expander("🧾 来源链接 & 思考过程（点击展开）", expanded=False):
        if sources:
            st.markdown("#### Sources（引用来源）")
            for s in sources:
                title = s.get("title") or s.get("url") or "来源"
                url = s.get("url") or ""
                tag = s.get("source_type") or ""
                prefix = f"[{tag}] " if tag else ""
                if url:
                    st.markdown(f"- {prefix}[{title}]({url})")
                else:
                    snippet = s.get("snippet") or ""
                    extra = f" · {snippet}" if snippet else ""
                    st.markdown(f"- {prefix}{title}{extra}")
        if steps:
            if sources:
                st.markdown("---")
            render_tool_steps_content(steps)


def render_tool_steps_content(steps: List[Any]) -> None:
    normalized = normalize_steps(steps)
    if not normalized:
        return

    st.markdown("#### 🧠 Agent 思考 / 工具调用过程")
    for idx, step in enumerate(normalized, start=1):
        tool_name = step.get("tool", "unknown_tool")
        st.markdown(f"**步骤 {idx} · `{tool_name}`**")
        st.markdown(f"- 工具输入：`{str(step.get('tool_input', ''))[:500]}`")

        observation = step.get("observation")
        if tool_name == "local_knowledge_search":
            st.markdown("- 检索到的本地文档片段（节选）：")
            docs = observation if isinstance(observation, list) else []
            for i, doc in enumerate(docs[:3], start=1):
                content = ""
                if isinstance(doc, dict):
                    content = doc.get("page_content") or doc.get("content") or ""
                snippet = (content or "").strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                st.markdown(f"  {i}. {snippet}")
        elif tool_name == "weibo_hot_search":
            st.markdown("- 微博实时热搜（节选）：")
            for item in parse_json_tool_items(observation, items_key="items")[:10]:
                if not isinstance(item, dict):
                    continue
                rank = item.get("rank", "?")
                title = item.get("title") or item.get("word") or ""
                heat = item.get("heat", "")
                tag = item.get("tag") or ""
                fetched = item.get("fetched_at") or ""
                tag_text = f" [{tag}]" if tag else ""
                st.markdown(f"  {rank}. {title}{tag_text}（热度 {heat}，抓取 {fetched}）")
        elif tool_name == "zhipu_web_search":
            st.markdown("- 智谱联网搜索结果（节选）：")
            for item in parse_json_tool_items(observation, items_key="items")[:8]:
                if not isinstance(item, dict):
                    continue
                rank = item.get("rank", "?")
                title = item.get("title") or ""
                url = item.get("url") or ""
                pub = item.get("publish_date") or ""
                snippet = str(item.get("content") or "").replace("\n", " ")[:120]
                st.markdown(f"  {rank}. [{title}]({url}) · {pub} · {snippet}")
        else:
            st.markdown("- 工具返回（节选）：")
            if isinstance(observation, list):
                for item in observation[:3]:
                    if isinstance(item, dict):
                        title = item.get("title") or "未提供标题"
                        url = item.get("url") or ""
                        snippet = item.get("content", "") or item.get("snippet", "")
                        snippet = str(snippet).replace("\n", " ")[:200]
                        st.markdown(f"  - **{title}** · {url} · {snippet}")
            elif isinstance(observation, dict) and observation.get("_type") == "raw":
                st.code(str(observation.get("content", ""))[:1000])
            else:
                st.code(str(observation)[:1000])


def render_chat_history() -> None:
    messages = st.session_state.get("messages", []) or []
    cid = int(st.session_state.get("conversation_id", 0))
    try:
        evals = api_client.get_evaluations(cid) if cid else []
    except Exception:
        evals = []
    eval_map = {(e["question"], e["answer"]): e for e in evals}

    for idx, m in enumerate(messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and idx > 0 and messages[idx - 1]["role"] == "user":
                key = (messages[idx - 1]["content"], m["content"])
                if key in eval_map:
                    render_eval_panel(eval_map[key])
                meta = m.get("metadata") or {}
                sources = meta.get("sources") or []
                steps = meta.get("steps") or []
                if sources or steps:
                    render_trace_panel(sources, steps)


def render_sidebar() -> bool:
    """渲染侧边栏。返回 API 是否可用。"""
    api_ok = True
    with st.sidebar:
        st.markdown("## ⚙️ Agent 设置")

        api_url = get_api_base_url()
        ok, msg = check_api_health(api_url)
        api_ok = ok
        if ok:
            st.success(f"🟢 后端已连接\n`{api_url}`")
        else:
            st.error(f"🔴 后端未就绪\n{msg}")
            return False

        is_generating = st.session_state.get("is_generating", False)
        cid = int(st.session_state["conversation_id"])

        st.markdown("### 📄 本地文档 RAG")
        uploaded = st.file_uploader(
            "上传 PDF/TXT 文档以启用本地知识检索",
            type=["pdf", "txt"],
            key="rag_file_uploader",
            disabled=is_generating,
        )

        if uploaded is not None and api_ok:
            upload_fingerprint = f"{cid}:{uploaded.name}:{uploaded.size}"
            already_ready = (
                st.session_state.get("rag_upload_fingerprint") == upload_fingerprint
                and st.session_state.get("rag_has_index")
            )
            if already_ready:
                st.success(f"当前文档：**{uploaded.name}** 已向量化，可直接提问。")
            else:
                with st.status("正在上传并向量化文档（API）...", expanded=True) as status:
                    step1 = st.empty()
                    step1.markdown("📤 正在上传文件到 API...")
                    try:
                        success, result = api_client.upload_document(
                            cid,
                            uploaded.name,
                            uploaded.getvalue(),
                        )
                    except Exception as e:
                        success, result = False, {"error": str(e)}

                    if not success:
                        status.update(label="文档解析失败", state="error")
                        err = result.get("error") or "上传失败"
                        step1.markdown(f"❌ {err}")
                        if result.get("warning"):
                            st.sidebar.warning(result["warning"])
                    else:
                        step1.markdown("✅ 文件上传与切片完成")
                        step2 = st.empty()
                        step2.markdown("✅ FAISS 向量库已在服务端构建")
                        st.session_state["rag_upload_fingerprint"] = upload_fingerprint
                        st.session_state["rag_has_index"] = True
                        if result.get("warning"):
                            st.sidebar.warning(result["warning"])
                        status.update(label="文档已成功载入并向量化 ✅", state="complete")

        if st.session_state.get("rag_has_index"):
            if not (
                uploaded is not None
                and st.session_state.get("rag_upload_fingerprint")
                == f"{cid}:{uploaded.name}:{uploaded.size}"
            ):
                st.success("本会话已启用本地文档检索（向量库在服务端）。")
        else:
            st.info("未上传文档，将主要依赖联网搜索。")

        st.markdown("---")
        st.markdown("### 🧵 会话管理")
        try:
            conversations = api_client.list_conversations()
        except Exception as e:
            st.error(f"加载会话列表失败：{e}")
            conversations = []

        if not conversations:
            created = api_client.create_conversation()
            st.session_state["conversation_id"] = created["id"]
            st.session_state["messages"] = api_client.get_messages(created["id"])
            conversations = api_client.list_conversations()

        def _label(c: Dict[str, Any]) -> str:
            if c.get("preview"):
                return f"{c['title']} · {c['preview']}"
            return c["title"]

        options = {_label(c): c["id"] for c in conversations}
        current_label = next(
            (lab for lab, _cid in options.items() if int(_cid) == cid),
            list(options.keys())[0],
        )
        selected_label = st.selectbox(
            "切换会话",
            list(options.keys()),
            index=list(options.keys()).index(current_label),
            disabled=is_generating,
        )
        selected_cid = int(options[selected_label])
        if selected_cid != cid:
            st.session_state["conversation_id"] = selected_cid
            st.session_state["messages"] = api_client.get_messages(selected_cid)
            st.session_state["rag_upload_fingerprint"] = None
            _refresh_rag_status()
            st.rerun()

        st.markdown("##### ✏️ 重命名会话")
        current_conv = next((c for c in conversations if int(c["id"]) == cid), None)
        current_title = current_conv["title"] if current_conv else "未命名会话"
        new_title = st.text_input("会话标题", value=current_title, disabled=is_generating)
        if st.button("💾 保存新标题", use_container_width=True, disabled=is_generating):
            safe = new_title.strip()
            if safe and safe != current_title:
                try:
                    api_client.rename_conversation(cid, safe)
                except Exception as e:
                    st.sidebar.error(f"重命名失败：{e}")

        st.markdown("##### ⚙️ 会话设置")
        cols = st.columns(2)
        with cols[0]:
            if st.button("➕ 开启新会话", use_container_width=True, disabled=is_generating):
                now = datetime.datetime.now()
                created = api_client.create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
                st.session_state["conversation_id"] = created["id"]
                st.session_state["messages"] = api_client.get_messages(created["id"])
                st.session_state["rag_upload_fingerprint"] = None
                st.session_state["rag_has_index"] = False
                st.rerun()
        with cols[1]:
            if st.button("删除会话", use_container_width=True, disabled=is_generating):
                try:
                    api_client.delete_conversation(cid)
                    remaining = api_client.list_conversations()
                    if remaining:
                        st.session_state["conversation_id"] = remaining[0]["id"]
                        st.session_state["messages"] = api_client.get_messages(remaining[0]["id"])
                    else:
                        created = api_client.create_conversation()
                        st.session_state["conversation_id"] = created["id"]
                        st.session_state["messages"] = api_client.get_messages(created["id"])
                    st.session_state["rag_upload_fingerprint"] = None
                    _refresh_rag_status()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"删除失败：{e}")

        st.markdown("---")
        st.markdown("### 🧪 回答质量评测")
        st.session_state.setdefault("enable_judge", True)
        st.session_state["enable_judge"] = st.toggle(
            "启用 Judge 自动打分",
            value=st.session_state["enable_judge"],
            disabled=is_generating,
        )
        from core.evaluator import list_available_judge_providers

        judge_options = list_available_judge_providers()
        if judge_options and st.session_state["enable_judge"]:
            provider_ids = [p["provider"] for p in judge_options]
            label_map = {p["provider"]: p["display"] for p in judge_options}
            default_provider = (st.session_state.get("judge_provider") or "").strip().lower()
            if default_provider not in provider_ids:
                import os

                env_default = (os.getenv("JUDGE_PROVIDER") or "").strip().lower()
                if env_default in provider_ids:
                    default_provider = env_default
                else:
                    default_provider = "zhipu" if "zhipu" in provider_ids else provider_ids[0]
            st.session_state["judge_provider"] = st.selectbox(
                "Judge 打分模型（建议与主回答不同厂，做交叉评测）",
                options=provider_ids,
                index=provider_ids.index(default_provider),
                format_func=lambda pid: label_map.get(pid, pid),
                disabled=is_generating,
            )
            if len(judge_options) >= 2:
                st.caption("主回答默认 DeepSeek，Judge 选智谱 = 跨模型质检。")
        elif st.session_state["enable_judge"]:
            st.caption("未检测到可用 Judge API Key，请在 .env 配置 DEEPSEEK 或 ZHIPU。")

        st.markdown("---")
        st.markdown("### 📡 实时数据")
        st.session_state.setdefault("enable_weibo_hot", True)
        st.session_state["enable_weibo_hot"] = st.toggle(
            "启用微博实时热搜",
            value=st.session_state["enable_weibo_hot"],
            disabled=is_generating,
        )
        st.session_state.setdefault("enable_zhipu_search", True)
        st.session_state["enable_zhipu_search"] = st.toggle(
            "启用智谱联网搜索",
            value=st.session_state["enable_zhipu_search"],
            disabled=is_generating,
        )
        st.session_state.setdefault("enable_forced_rag", True)
        st.session_state["enable_forced_rag"] = st.toggle(
            "启用文档强制检索",
            value=st.session_state["enable_forced_rag"],
            disabled=is_generating,
        )

        st.markdown("---")
        if st.button("🗑️ 清除所有历史记录", use_container_width=True, disabled=is_generating):
            try:
                api_client.clear_all_conversations()
                created = api_client.create_conversation()
                st.session_state.clear()
                st.session_state["conversation_id"] = created["id"]
                st.session_state["messages"] = api_client.get_messages(created["id"])
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"清除失败：{e}")

    return api_ok


def render_quick_start_buttons() -> None:
    has_user_msg = any(m.get("role") == "user" for m in (st.session_state.get("messages") or []))
    if has_user_msg:
        return
    st.markdown("#### 🚀 快捷开始")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📄 总结这份文档", use_container_width=True, disabled=st.session_state.get("is_generating", False)):
            st.session_state["pending_user_input"] = "请总结这份文档/论文的核心内容，并用要点列出主要贡献与结论。"
            st.rerun()
    with c2:
        if st.button("🔥 今天微博热搜前三", use_container_width=True, disabled=st.session_state.get("is_generating", False)):
            today = datetime.datetime.now()
            st.session_state["pending_user_input"] = (
                f"请调用微博实时接口，列出 {today:%Y年%m月%d日} 热搜榜前三名及背景。"
            )
            st.rerun()
    with c3:
        if st.button("🤖 介绍一下 DeepSeek", use_container_width=True, disabled=st.session_state.get("is_generating", False)):
            st.session_state["pending_user_input"] = (
                "请介绍一下 DeepSeek 的产品与模型能力、典型应用场景，以及与同类模型相比的特点。"
            )
            st.rerun()
