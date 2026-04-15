import os
import datetime
import tempfile
import sqlite3
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")


def init_db() -> None:
    """初始化 SQLite 数据库、会话表、消息表，并做简单的向后兼容迁移。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # 会话表（如不存在则创建）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # 消息表（如不存在则创建，包含 conversation_id）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        # 针对旧版本 messages 表做字段迁移：如果没有 conversation_id 列，则添加
        cur.execute("PRAGMA table_info(messages)")
        columns_info = cur.fetchall()
        column_names = {row[1] for row in columns_info}  # row[1] 是列名
        if "conversation_id" not in column_names:
            # 按用户要求，使用 TEXT 类型以兼容旧表
            cur.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")

        # 确保索引存在（若旧表没有该列，前面已添加）
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id)"
        )

        conn.commit()
    finally:
        conn.close()

def list_conversations() -> List[Dict[str, Any]]:
    """列出所有会话（按最近创建排序），并附带最后一条消息预览。"""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.id DESC
                    LIMIT 1
                ) AS last_message
            FROM conversations c
            ORDER BY c.id DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "last_message": r[3] or "",
        }
        for r in rows
    ]


def create_conversation(title: str) -> int:
    """新建会话并返回会话 ID。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def delete_all_conversations_and_messages() -> None:
    """清空所有会话与消息。"""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM conversations")
        conn.commit()
    finally:
        conn.close()


def update_conversation_title(conversation_id: int, new_title: str) -> None:
    """更新指定会话标题。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (new_title, int(conversation_id)),
        )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> None:
    """删除单个会话及其全部消息。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (int(conversation_id),))
        cur.execute("DELETE FROM conversations WHERE id = ?", (int(conversation_id),))
        conn.commit()
    finally:
        conn.close()


def load_history_from_db(conversation_id: int) -> List[Dict[str, str]]:
    """从 SQLite 读取指定会话的历史聊天记录。"""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [{"role": r, "content": c} for r, c in rows]


def save_message_to_db(conversation_id: int, role: str, content: str) -> None:
    """将一条消息写入 SQLite。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def init_env() -> None:
    """加载环境变量。"""
    load_dotenv()


def get_llm() -> ChatOpenAI:
    """创建 DeepSeek Chat LLM。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DEEPSEEK_API_KEY，请在 .env 中配置。")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.3,
    )
    return llm


def get_tools() -> List[Any]:
    """创建 Tavily 搜索工具。"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("未检测到 TAVILY_API_KEY，请在 .env 中配置。")

    tavily_search = TavilySearchResults(
        max_results=3,
    )
    return [tavily_search]


@st.cache_resource(show_spinner=False)
def get_agent_executor() -> AgentExecutor:
    """构建带工具调用能力的 AgentExecutor。"""
    init_env()
    llm = get_llm()
    tools = get_tools()

    # 获取当前真实时间，格式化为日期 + 时间 + 星期几
    now = datetime.datetime.now()
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_cn = weekday_map[now.weekday()]
    current_time_str = (
        f"当前本机系统时间为：{now:%Y-%m-%d %H:%M:%S}（星期{weekday_cn}，24 小时制）。"
    )

    # 提示词中说明文档优先 + 何时使用实时搜索
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{current_time_str}\n"
                    f"你是一个具备实时联网能力的超级智能体。今天是 {now:%Y-%m-%d}（星期{weekday_cn}）。\n"
                    "当用户询问“今天”“最近”“热点”“新闻”或任何具有时效性的问题时，你必须、绝对要优先调用 Tavily 搜索工具去获取最新的网页信息，绝不能依赖你的历史固有知识瞎编！\n\n"
                    "当前会话中，用户可能上传了一份本地文档用于问答。如果下面提供了 `文档上下文`，你必须 **优先严格依据该文档内容** 来回答与文档相关的问题：\n"
                    "1. 当问题与文档内容相关时，尽量从文档中查找并归纳答案，必要时可进行合理推理，但不要凭空捏造文档中不存在的信息。\n"
                    "2. 如果文档中明显不包含足够信息，且问题又具有时效性或需要外部知识支持，可以再调用 Tavily 搜索工具获取最新网页信息进行补充。\n"
                    "3. 回答时清晰标注哪些是来自文档的依据，哪些是通过联网搜索得到的补充结论。\n\n"
                    "【文档上下文】（可能被截断，仅供参考）：\n"
                    "{doc_context}\n"
                    "—— 文档上下文结束 ——\n\n"
                    "回答时请使用清晰、结构化的中文表述；如果使用了搜索工具，请在回答末尾列出使用到的主要链接（Sources）。"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return agent_executor


def init_session_state() -> None:
    """初始化会话状态。"""
    # 当前会话 ID
    if "active_conversation_id" not in st.session_state:
        conversations = list_conversations()
        if conversations:
            st.session_state["active_conversation_id"] = conversations[0]["id"]
        else:
            now = datetime.datetime.now()
            new_id = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
            st.session_state["active_conversation_id"] = new_id

    if "messages" not in st.session_state:
        cid = int(st.session_state["active_conversation_id"])
        history = load_history_from_db(cid)
        if history:
            st.session_state["messages"] = history
        else:
            welcome_msg = {
                "role": "assistant",
                "content": "你好，我是你的 AI 联网搜索与本地文档问答助手。可以上传 PDF/TXT 文档并围绕其提问，"
                "我也会在需要时自动联网搜索最新信息。",
            }
            st.session_state["messages"] = [welcome_msg]
            save_message_to_db(cid, welcome_msg["role"], welcome_msg["content"])
    if "doc_context" not in st.session_state:
        st.session_state["doc_context"] = ""
    if "doc_meta" not in st.session_state:
        st.session_state["doc_meta"] = {}

def switch_conversation(conversation_id: int) -> None:
    """切换会话：更新 active_conversation_id，并从 DB 载入消息。"""
    st.session_state["active_conversation_id"] = int(conversation_id)
    st.session_state["messages"] = load_history_from_db(int(conversation_id))
    # 文档上下文按“会话内”暂存；切换会话时清空，避免串文档
    st.session_state["doc_context"] = ""
    st.session_state["doc_meta"] = {}


def render_chat_history() -> None:
    """渲染历史消息。"""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def convert_history_for_agent() -> List[Dict[str, str]]:
    """将 Streamlit 的消息格式转换为 Agent 可使用的历史记录格式。"""
    history: List[Dict[str, str]] = []
    for msg in st.session_state.get("messages", []):
        role = msg["role"]
        if role == "user":
            mapped_role = "human"
        elif role == "assistant":
            mapped_role = "ai"
        else:
            mapped_role = role
        history.append({"role": mapped_role, "content": msg["content"]})
    return history


def extract_sources_from_intermediate_steps(
    intermediate_steps: List[Any], tavily_tool_name: str
) -> List[Dict[str, str]]:
    """从中间步骤中提取 Tavily 返回的链接与标题。"""
    results: List[Dict[str, str]] = []
    seen_urls = set()
    for action, observation in intermediate_steps:
        if getattr(action, "tool", "") != tavily_tool_name:
            continue

        if isinstance(observation, list):
            for item in observation:
                if isinstance(item, dict):
                    url = item.get("url")
                    title = item.get("title") or "未提供标题"
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({"url": url, "title": title})
    return results


def parse_uploaded_document() -> None:
    """解析用户上传的文档，并将上下文写入 session_state。"""
    st.sidebar.markdown("### 📄 本地文档问答")
    uploaded_file = st.sidebar.file_uploader(
        "上传 PDF 或 TXT 文件，用于基于文档的问答",
        type=["pdf", "txt"],
    )

    if uploaded_file is None:
        # 不强制清空，方便用户保留上一份文档上下文
        if st.session_state.get("doc_context"):
            st.sidebar.info("当前仍在使用上一份已解析的文档上下文。")
        else:
            st.sidebar.info("尚未上传任何文档，将仅使用联网搜索回答问题。")
        return

    # 解析文件
    doc_text = ""
    file_name = uploaded_file.name

    try:
        if file_name.lower().endswith(".pdf"):
            # 将上传内容写入临时文件，再用 PyPDFLoader 解析
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            doc_text = "\n".join(page.page_content for page in pages)
        else:
            # 认为是纯文本
            raw_bytes = uploaded_file.getvalue()
            doc_text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        st.sidebar.error(f"文档解析失败：{e}")
        return

    if not doc_text.strip():
        st.sidebar.warning("文档内容为空或无法提取文字。")
        return

    # 截断长度，避免过长
    max_chars = 10000
    truncated_text = doc_text[:max_chars]

    st.session_state["doc_context"] = truncated_text
    st.session_state["doc_meta"] = {
        "file_name": file_name,
        "char_count": len(truncated_text),
    }

    st.sidebar.success("文档已解析完毕，您可以开始针对文档提问了。")
    st.sidebar.caption(
        f"当前文档：`{file_name}`，用于提示的文本长度约为 {len(truncated_text)} 字符（已自动截断）。"
    )


def main() -> None:
    st.set_page_config(page_title="DeepSearch Agent", page_icon="🔍", layout="wide")
    # 初始化数据库和会话
    init_db()
    init_session_state()

    # --- 侧边栏：Agent 设置 + 文档上传 ---
    with st.sidebar:
        st.markdown("## ⚙️ Agent 设置")

        # 会话管理：新建/切换
        st.markdown("### 🧵 会话管理")
        conversations = list_conversations()
        if not conversations:
            now = datetime.datetime.now()
            new_id = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
            conversations = list_conversations()
            switch_conversation(new_id)

        # 构造带“最后一条消息预览”的会话选项
        def _build_label(c: Dict[str, Any]) -> str:
            preview = (c["last_message"] or "").replace("\n", " ").strip()
            if len(preview) > 30:
                preview = preview[:30] + "..."
            if preview:
                return f"{c['title']} · {preview}"
            return f"{c['title']}"

        options = {_build_label(c): c["id"] for c in conversations}
        current_cid = int(st.session_state.get("active_conversation_id", conversations[0]["id"]))
        current_label = None
        for label, cid in options.items():
            if int(cid) == current_cid:
                current_label = label
                break
        is_generating = st.session_state.get("is_generating", False)
        selected_label = st.selectbox(
            "切换会话",
            list(options.keys()),
            index=list(options.keys()).index(current_label) if current_label in options else 0,
            disabled=is_generating,
        )
        selected_cid = int(options[selected_label])
        if selected_cid != current_cid:
            switch_conversation(selected_cid)
            st.rerun()

        # 重命名当前会话
        st.markdown("##### ✏️ 重命名当前会话")
        # 取当前会话原始标题
        current_conversation = next(
            (c for c in conversations if int(c["id"]) == current_cid),
            None,
        )
        current_title = current_conversation["title"] if current_conversation else "未命名会话"
        new_title = st.text_input(
            "会话标题",
            value=current_title,
            key="conversation_title_input",
            disabled=is_generating,
        )
        if st.button("💾 保存新标题", use_container_width=True, disabled=is_generating):
            safe_title = new_title.strip()
            if safe_title and safe_title != current_title:
                update_conversation_title(current_cid, safe_title)

        # 会话设置：删除 / 新建
        st.markdown("##### ⚙️ 会话设置")
        if st.button("删除当前会话", use_container_width=True, disabled=is_generating):
            delete_conversation(current_cid)
            # 重新选择一个会话或新建一个
            remaining = list_conversations()
            if remaining:
                switch_conversation(remaining[0]["id"])
            else:
                now = datetime.datetime.now()
                new_id = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
                switch_conversation(new_id)
            st.rerun()

        if st.button("➕ 开启新会话", use_container_width=True, disabled=is_generating):
            now = datetime.datetime.now()
            new_id = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
            switch_conversation(new_id)
            st.rerun()

        st.markdown("---")

        # 清除记忆按钮
        if st.button("🗑️ 清除所有对话记忆", use_container_width=True):
            delete_all_conversations_and_messages()
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        parse_uploaded_document()

    # --- 主区域标题与简介 ---
    st.title("🔍 DeepSearch Agent")
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0; color: #4a4a4a;">
        集成 <b>DeepSeek</b> 大模型、<b>Tavily</b> 联网搜索 与 <b>本地文档（RAG-like）</b> 的多模态智能体。<br/>
        · 支持实时热点、新闻查询<br/>
        · 支持上传 PDF/TXT 文档进行深度问答<br/>
        · 自动在“本地文档优先”与“联网检索”之间切换
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 欢迎页：无用户对话时显示快捷提示词 ---
    has_user_msg = any(m["role"] == "user" for m in st.session_state["messages"])
    col_main, col_side = st.columns([3, 2])

    with col_main:
        if not has_user_msg:
            st.markdown(
                """
                <div style="margin-top: 3rem; text-align: center;">
                  <h2>👋 欢迎使用 DeepSearch Agent</h2>
                  <p style="color: #666; font-size: 0.95rem;">
                    你可以像和人类助手聊天一样，向我提出任何问题。<br/>
                    也可以先上传一份 PDF/TXT 文档，我会围绕文档内容进行专业解读与问答。
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 🚀 快速体验")
            c1, c2, c3 = st.columns(3)
            examples = [
                "帮我查一下今天 A 股大盘的整体走势和主要指数表现。",
                "总结一下最近一周全球 AI 领域的重点新闻和趋势。",
                "我上传了一份文档，请帮我先用要点式总结一下它的核心内容。",
            ]
            example_clicked = None
            with c1:
                if st.button("📈 今天 A 股大盘", use_container_width=True):
                    example_clicked = examples[0]
            with c2:
                if st.button("📰 最近 AI 热点", use_container_width=True):
                    example_clicked = examples[1]
            with c3:
                if st.button("📚 总结上传文档", use_container_width=True):
                    example_clicked = examples[2]

            if example_clicked:
                st.session_state.setdefault("pending_user_input", example_clicked)
                st.rerun()

    # --- 聊天历史区域 ---
    with col_main:
        render_chat_history()

    agent_executor = get_agent_executor()
    tavily_tool_name = ""
    for tool in agent_executor.tools:
        if isinstance(tool, TavilySearchResults):
            tavily_tool_name = tool.name
            break

    # 模式提示放在副列
    with col_side:
        if st.session_state.get("doc_context"):
            meta = st.session_state.get("doc_meta", {})
            file_name = meta.get("file_name", "已上传文档")
            st.markdown("#### 📂 当前文档状态")
            st.success(
                f"已加载文档：`{file_name}`\n\n"
                "我会 **优先基于该文档内容回答问题**，在文档无相关信息时再考虑联网搜索。"
            )
        else:
            st.markdown("#### 📂 当前文档状态")
            st.info("尚未加载任何文档，我将主要依赖自身知识并在需要时自动联网搜索。")

        st.markdown("---")
        st.markdown("#### 💡 小提示")
        st.caption(
            "· 使用“今天 / 最近 / 热点 / 新闻”等关键词时，我会自动走联网搜索。\n"
            "· 针对已上传文档提问时，我会优先只基于文档内容进行回答。"
        )

    # --- 用户输入（包含快捷提示词触发） ---
    default_input = st.session_state.pop("pending_user_input", "")
    user_input = st.chat_input(
        "请输入你的问题，例如：针对文档的某一章节进行提问，或询问今天的科技热点。",
    )
    if not user_input and default_input:
        # 快捷按钮触发
        user_input = default_input

    if not user_input:
        return

    # 记录用户消息
    st.session_state["messages"].append({"role": "user", "content": user_input})
    active_cid = int(st.session_state["active_conversation_id"])
    save_message_to_db(active_cid, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- 助手消息（流式输出 + 思考过程可视化） ---
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        thinking_placeholder = st.empty()

        chat_history_for_agent = convert_history_for_agent()
        doc_context = st.session_state.get("doc_context", "") or ""

        # 标记为正在生成，避免在这一轮中途切换/删除会话
        st.session_state["is_generating"] = True

        # 流式调用 AgentExecutor.stream
        final_state: Dict[str, Any] = {}
        full_answer_text = ""

        for step in agent_executor.stream(
            {
                "input": user_input,
                "chat_history": chat_history_for_agent,
                "doc_context": doc_context,
            }
        ):
            final_state = step
            chunk = step.get("output", "")
            if isinstance(chunk, str) and chunk:
                full_answer_text = chunk
                answer_placeholder.markdown(full_answer_text)

        # 获取中间工具调用信息
        intermediate_steps: List[Any] = final_state.get("intermediate_steps", [])
        used_sources: List[Dict[str, str]] = []

        if intermediate_steps and tavily_tool_name:
            sources = extract_sources_from_intermediate_steps(
                intermediate_steps, tavily_tool_name
            )
            if sources:
                used_sources = sources
                with thinking_placeholder.container():
                    with st.status(
                        "Agent 正在全网检索信息（Tavily）...", expanded=True
                    ) as status:
                        if doc_context:
                            st.write(
                                "文档信息可能不足以完整回答你的问题，我正在 **结合 Tavily 搜索最新网页信息** 进行补充。"
                            )
                        else:
                            st.write(
                                "检测到你的问题具有较强的时效性，我已调用 Tavily 搜索工具进行联网检索。"
                            )
                        st.write("以下为部分检索到的网页：")
                        for item in sources:
                            st.markdown(
                                f"- [{item['title']}]({item['url']})",
                                unsafe_allow_html=False,
                            )
                        status.update(label="检索完成 ✅", state="complete")

                    with st.expander("🧠 查看 Agent 的检索与思考过程", expanded=False):
                        st.markdown("**Tavily 搜索到的主要网页：**")
                        for item in sources:
                            st.markdown(
                                f"- [{item['title']}]({item['url']})",
                                unsafe_allow_html=False,
                            )

        # 在回答末尾附上 Sources
        if used_sources:
            sources_md_lines = [
                f"- [{item['title']}]({item['url']})" for item in used_sources
            ]
            full_answer_text = (
                full_answer_text
                + "\n\n---\n\n**Sources（部分参考链接）**:\n"
                + "\n".join(sources_md_lines)
            )

        answer_placeholder.markdown(full_answer_text)

    # 将最终回答写入会话历史
    st.session_state["is_generating"] = False
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": full_answer_text,
        }
    )
    save_message_to_db(active_cid, "assistant", full_answer_text)


if __name__ == "__main__":
    main()

