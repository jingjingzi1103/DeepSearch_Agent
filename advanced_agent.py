import os
import sqlite3
import tempfile
import datetime
import time
import shutil
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


###########################################################################
# 一、环境变量与数据库初始化
###########################################################################

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")
FAISS_ROOT_DIR = os.path.join(os.path.dirname(__file__), "faiss_index_store")


def get_faiss_dir_for_conversation(conversation_id: int) -> str:
    """根据会话 ID 生成对应的 FAISS 向量库目录。"""
    return os.path.join(FAISS_ROOT_DIR, f"conv_{int(conversation_id)}")


def delete_faiss_dir_for_conversation(conversation_id: int) -> None:
    """删除指定会话对应的 FAISS 向量库目录（如存在）。"""
    path = get_faiss_dir_for_conversation(conversation_id)
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"[FAISS] removed index dir for conversation {conversation_id}: {path}")
        except Exception as e:
            print(f"[FAISS] failed to remove index dir {path}: {e}")


def init_env() -> None:
    """加载 .env 中的环境变量。

    LangSmith 链路追踪（预留）：
    - LANGCHAIN_TRACING_V2: "true"/"false"
    - LANGCHAIN_API_KEY: LangSmith API Key
    这里只做检测与提示，不做额外集成代码。
    """
    load_dotenv()

    tracing_v2 = (os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    langchain_api_key = (os.getenv("LANGCHAIN_API_KEY") or "").strip()
    enabled = tracing_v2 in {"1", "true", "yes", "on"} and bool(langchain_api_key)
    print(f"[LangSmith] tracing_v2 enabled: {enabled}")


def init_db() -> None:
    """初始化本地 SQLite 数据库与消息表。

    表结构：
    - messages:
        - id: 自增主键
        - conversation_id: 会话编号（整数，方便支持多会话）
        - role: 'user' / 'assistant'
        - content: 文本内容
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # 会话元数据表：用于保存会话标题（UI 不展示 ID，但后台仍需要区分会话）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # 兼容迁移：旧库可能没有 conversation_id 列，或该列存在但为 NULL
        cur.execute("PRAGMA table_info(messages)")
        cols = {row[1] for row in cur.fetchall()}
        if "conversation_id" not in cols:
            cur.execute("ALTER TABLE messages ADD COLUMN conversation_id INTEGER")

        # 若历史数据存在但 conversation_id 为空，则统一归入默认会话 1
        # 同时确保至少存在一个会话元数据
        cur.execute("SELECT COUNT(1) FROM conversations")
        conv_cnt = cur.fetchone()[0]
        if not conv_cnt:
            cur.execute("INSERT INTO conversations (title) VALUES (?)", ("默认会话",))

        cur.execute("UPDATE messages SET conversation_id = 1 WHERE conversation_id IS NULL")

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id)"
        )
        conn.commit()
    finally:
        conn.close()

def list_conversations_with_preview() -> List[Dict[str, Any]]:
    """列出会话（title + 最后一条消息预览），用于侧边栏选择。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                c.id,
                c.title,
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
    out: List[Dict[str, Any]] = []
    for cid, title, last_msg in rows:
        preview = (last_msg or "").replace("\n", " ").strip()
        if len(preview) > 30:
            preview = preview[:30] + "..."
        out.append({"id": int(cid), "title": title, "preview": preview})
    return out


def create_conversation(title: str) -> int:
    """创建新会话并返回会话 ID。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def update_conversation_title(conversation_id: int, new_title: str) -> None:
    """重命名会话。"""
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
    """删除单个会话及其消息。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE conversation_id = ?", (int(conversation_id),))
        cur.execute("DELETE FROM conversations WHERE id = ?", (int(conversation_id),))
        conn.commit()
    finally:
        conn.close()
    # 删除该会话对应的 FAISS 索引目录
    delete_faiss_dir_for_conversation(conversation_id)


def _get_next_conversation_id() -> int:
    """从数据库中计算下一个会话编号。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(conversation_id) FROM messages")
        row = cur.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        return int(max_id) + 1
    finally:
        conn.close()


def load_history_from_db(conversation_id: int) -> List[Dict[str, str]]:
    """按会话编号加载历史消息列表。"""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [{"role": r, "content": c} for r, c in rows]


def save_message_to_db(conversation_id: int, role: str, content: str) -> None:
    """将一条消息写入数据库。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


def clear_all_history() -> None:
    """清空所有历史记录。"""
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
    # 同时清空所有 FAISS 索引目录
    if os.path.isdir(FAISS_ROOT_DIR):
        try:
            shutil.rmtree(FAISS_ROOT_DIR)
            print(f"[FAISS] cleared all index dirs under {FAISS_ROOT_DIR}")
        except Exception as e:
            print(f"[FAISS] failed to clear FAISS_ROOT_DIR {FAISS_ROOT_DIR}: {e}")


###########################################################################
# 二、LLM 与 Embedding 引擎初始化
###########################################################################


def get_llm() -> ChatOpenAI:
    """创建 DeepSeek Chat LLM 客户端。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未检测到 DEEPSEEK_API_KEY，请在 .env 中配置。")

    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.3,
    )


def get_embedding_model() -> OpenAIEmbeddings:
    """创建硅基流动 BGE-M3 Embedding 客户端。"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("未检测到 SILICONFLOW_API_KEY，请在 .env 中配置。")

    return OpenAIEmbeddings(
        openai_api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        model="BAAI/bge-m3",
        # 硅基流动对 embedding 的 batch size 有上限（常见为 64）。
        # 如果一次性切片过多（例如 193 个 chunk），不限制批量会触发 413：
        # "input batch size X > maximum allowed batch size 64"
        chunk_size=64,
    )


###########################################################################
# 三、RAG：文档上传、切片、向量化与检索工具
###########################################################################


def build_vector_store_from_file(uploaded_file, conversation_id: int) -> Optional[FAISS]:
    """根据用户上传的文件构建 FAISS 向量库。

    - 支持 PDF：使用 PyPDFLoader 逐页加载
    - 支持 TXT：直接按 UTF-8 解码
    - 使用 RecursiveCharacterTextSplitter 做切片
    - 使用硅基流动 BGE-M3 做向量化
    """
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()
    raw_text = ""

    try:
        if file_name.endswith(".pdf"):
            # 将上传内容写入临时文件，再用 PyPDFLoader 解析
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            raw_text = "\n".join(page.page_content for page in pages)
        elif file_name.endswith(".txt"):
            raw_bytes = uploaded_file.getvalue()
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
        else:
            st.sidebar.error("目前仅支持 PDF 和 TXT 文件。")
            return None
    except Exception as e:
        st.sidebar.error(f"解析文档时出错：{e}")
        return None

    if not raw_text.strip():
        st.sidebar.warning("文档中未提取到有效文本内容。")
        return None

    # 使用递归切片器做“真切片”，避免长文档直接塞入 Prompt
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )
    docs = splitter.create_documents([raw_text])

    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(docs, embeddings)
        # 向量库落盘：按会话 ID 分目录，避免不同对话互相污染
        index_dir = get_faiss_dir_for_conversation(conversation_id)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        return vector_store
    except Exception as e:
        # 典型报错：413 - input batch size 193 > maximum allowed batch size 64
        # 解决：降低 embeddings 的 chunk_size（本文件中已设为 64）
        st.sidebar.error(f"构建向量库失败：{e}")
        return None


def build_retriever_tool(vector_store: FAISS):
    """将向量库包装为 LangChain 的 Retriever 工具。"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    tool = create_retriever_tool(
        retriever=retriever,
        name="local_knowledge_search",
        description=(
            "基于用户上传的本地文档进行语义检索，"
            "适合回答与该文档内容紧密相关的问题，例如“根据文档内容总结第3章”之类。"
        ),
    )
    return tool


###########################################################################
# 四、工具集合与 Agent 构建
###########################################################################


def get_tavily_tool() -> TavilySearchResults:
    """创建 Tavily 联网搜索工具。"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("未检测到 TAVILY_API_KEY，请在 .env 中配置。")

    return TavilySearchResults(max_results=5)


def build_agent(tools: List[Any]) -> AgentExecutor:
    """根据给定工具列表构建多工具 AgentExecutor。"""
    llm = get_llm()

    # 系统提示：强调实时日期与工具优先
    now = datetime.datetime.now()
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_cn = weekday_map[now.weekday()]
    current_time_str = f"今天是 {now:%Y-%m-%d}（星期{weekday_cn}），当前时间 {now:%H:%M:%S}。"

    # 判断当前是否启用了本地文档检索工具（启用则代表：用户已上传文档且向量库可用）
    has_local_tool = any(getattr(t, "name", "") == "local_knowledge_search" for t in tools)
    local_tool_hard_rule = ""
    if has_local_tool:
        local_tool_hard_rule = (
            "\n【本地文档检索硬规则（必须遵守）】\n"
            "当前工具列表中包含 `local_knowledge_search`，这意味着用户已上传文档且系统已完成向量化。\n"
            "因此：\n"
            "1) 你绝对不允许说“我没看到你上传文档/你还没上传文档/我无法访问文档”等类似话术。\n"
            "2) 只要用户的问题与文档/论文/这篇/这份/根据文档/摘要/结论/方法/实验/图表/表格/章节/附录等相关，\n"
            "   你必须先调用 `local_knowledge_search` 至少 1 次获取文档片段，再基于片段回答。\n"
            "3) 若 `local_knowledge_search` 返回的片段不足以回答，再视情况调用 Tavily；否则不要联网。\n"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    f"{current_time_str}\n"
                    "你是一名商业级 AI 智能体，具备联网搜索与本地知识检索能力。\n"
                    "必须遵守以下原则：\n"
                    "1. 对任何问题，都要优先考虑调用工具获取“真实上下文”后再回答，"
                    "   尤其是涉及实时信息（今天、最近、热点、新闻、股价等）或与上传文档紧密相关的问题。\n"
                    "2. 当使用工具返回了上下文时，回答必须基于这些上下文进行归纳与推理，"
                    "   不得编造上下文中不存在的具体事实。\n"
                    "3. 如果工具未返回足够信息，请诚实说明“不确定”或“文档/搜索结果中未找到”，"
                    "   而不是凭空瞎编。\n"
                    "4. 回答风格要求：清晰、结构化、条理分明，重要结论可以使用列表或小标题呈现。\n"
                    f"{local_tool_hard_rule}"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor


###########################################################################
# 五、Streamlit 状态初始化与工具函数
###########################################################################


def init_session_state() -> None:
    """初始化 Streamlit 的会话状态。"""
    # 当前会话编号：优先选择最新会话；没有则创建默认会话
    if "conversation_id" not in st.session_state:
        conversations = list_conversations_with_preview()
        if conversations:
            st.session_state["conversation_id"] = conversations[0]["id"]
        else:
            now = datetime.datetime.now()
            st.session_state["conversation_id"] = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")

    # 聊天消息列表
    if "messages" not in st.session_state:
        cid = int(st.session_state["conversation_id"])
        history = load_history_from_db(cid)
        if history:
            st.session_state["messages"] = history
        else:
            welcome = {
                "role": "assistant",
                "content": (
                    "你好，我是你的多引擎 AI Agent：支持 DeepSeek 联网搜索 + 本地文档 RAG 检索。\n"
                    "左侧可以上传 PDF/TXT 文档，我会基于文档和网络信息为你提供专业回答。"
                ),
            }
            st.session_state["messages"] = [welcome]
            save_message_to_db(cid, welcome["role"], welcome["content"])

    # 文档向量库与工具
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("local_tool", None)
    st.session_state.setdefault("rag_index_loaded_from_disk", False)
    # 若本地存在已落盘的 FAISS 向量库，则在启动时自F动加载（避免刷新丢失）
    index_dir = os.path.join(os.path.dirname(__file__), "faiss_index_store")
    if st.session_state.get("vector_store") is None and os.path.isdir(index_dir):
        try:
            embeddings = get_embedding_model()
            vector_store = FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            st.session_state["vector_store"] = vector_store
            st.session_state["local_tool"] = build_retriever_tool(vector_store)
            st.session_state["rag_index_loaded_from_disk"] = True
        except Exception as e:
            # 加载失败不阻断应用启动，用户仍可重新上传构建
            print(f"[FAISS] load_local failed: {e}")

    # 标记当前是否在生成回答，避免中途切换会话状态打断流
    st.session_state.setdefault("is_generating", False)

    # 快捷提示词（欢迎页按钮触发时写入）
    st.session_state.setdefault("pending_user_input", "")


def convert_history_for_agent() -> List[Dict[str, str]]:
    """将历史消息转换为 LangChain 可用格式（带滑动窗口截断）。

    为防止对话过长导致上下文/Token 超限：
    - 若历史消息超过 10 条，仅保留最近 10 条（约最近 5 轮对话）
    """
    history: List[Dict[str, str]] = []
    messages = st.session_state.get("messages", []) or []
    if len(messages) > 10:
        messages = messages[-10:]

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


def render_chat_history() -> None:
    """逐条渲染历史聊天记录。"""
    for m in st.session_state.get("messages", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


###########################################################################
# 六、主界面与交互逻辑
###########################################################################


def render_sidebar() -> None:
    """渲染侧边栏：会话控制 + 文档上传。"""
    with st.sidebar:
        st.markdown("## ⚙️ Agent 设置")

        is_generating = st.session_state.get("is_generating", False)
        cid = int(st.session_state["conversation_id"])

        # 确保当前会话的向量库状态与本地目录一致：
        # 若当前会话还没有加载向量库，但本地存在对应目录，则尝试即时加载。
        index_dir_for_cid = get_faiss_dir_for_conversation(cid)
        if (
            not is_generating
            and st.session_state.get("vector_store") is None
            and os.path.isdir(index_dir_for_cid)
        ):
            try:
                embeddings = get_embedding_model()
                vector_store = FAISS.load_local(
                    index_dir_for_cid,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                st.session_state["vector_store"] = vector_store
                st.session_state["local_tool"] = build_retriever_tool(vector_store)
                st.session_state["rag_index_loaded_from_disk"] = True
                print(f"[FAISS] lazy-loaded index for conversation {cid} from {index_dir_for_cid}")
            except Exception as e:
                print(f"[FAISS] lazy load failed for {index_dir_for_cid}: {e}")

        # 把文档上传放在侧边栏更靠上位置，避免被会话控件挤到下方看不到
        st.markdown("### 📄 本地文档 RAG")
        uploaded = st.file_uploader(
            "上传 PDF/TXT 文档以启用本地知识检索",
            type=["pdf", "txt"],
            key="rag_file_uploader",
            disabled=is_generating,
        )

        # 当用户上传新文档时，重建向量库与工具
        if uploaded is not None:
            with st.status("正在解析并向量化文档...", expanded=True) as status:
                st.write("📥 正在读取文件内容并切片（chunking）...")
                vector_store = build_vector_store_from_file(uploaded, conversation_id=cid)
                if vector_store is None:
                    status.update(label="文档解析失败", state="error")
                else:
                    st.write("🧠 正在构建 FAISS 向量库并创建检索工具...")
                    local_tool = build_retriever_tool(vector_store)
                    st.session_state["vector_store"] = vector_store
                    st.session_state["local_tool"] = local_tool
                    status.update(label="文档已成功载入并向量化 ✅", state="complete")

        if st.session_state.get("local_tool") is not None:
            if st.session_state.get("rag_index_loaded_from_disk"):
                st.success("已从本地缓存自动加载本会话的文档向量库，本地文档检索已启用。")
                st.caption("如果需要更换文档，可以重新上传新的 PDF/TXT 文件。")
            else:
                st.success("本会话的本地文档检索已启用：我会优先使用 local_knowledge_search 回答与文档相关的问题。")
        else:
            # 仅当“本会话对应的索引目录存在但加载失败”时才给出警告
            if os.path.isdir(index_dir_for_cid):
                st.warning("检测到本会话对应的向量库目录存在，但未成功加载。可以尝试重新上传文档以覆盖重建。")
            else:
                st.info("当前会话未加载任何本地文档，我会主要依赖 Tavily 联网搜索与模型自身知识。")

        st.markdown("---")
        st.markdown("### 🧵 会话管理")
        conversations = list_conversations_with_preview()
        if not conversations:
            now = datetime.datetime.now()
            cid = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
            st.session_state["conversation_id"] = cid
            st.session_state["messages"] = load_history_from_db(cid)
            conversations = list_conversations_with_preview()

        def _label(c: Dict[str, Any]) -> str:
            if c["preview"]:
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
            st.session_state["messages"] = load_history_from_db(selected_cid)
            # 切换会话时清空本地向量库，避免串文档
            st.session_state["vector_store"] = None
            st.session_state["local_tool"] = None
            st.session_state["rag_index_loaded_from_disk"] = False
            st.rerun()

        st.markdown("##### ✏️ 重命名会话")
        current_conv = next((c for c in conversations if int(c["id"]) == cid), None)
        current_title = current_conv["title"] if current_conv else "未命名会话"
        new_title = st.text_input("会话标题", value=current_title, disabled=is_generating)
        if st.button("💾 保存新标题", use_container_width=True, disabled=is_generating):
            safe = new_title.strip()
            if safe and safe != current_title:
                update_conversation_title(cid, safe)
                # 不强制 rerun：避免打断正在生成；用户下一次操作自然刷新

        st.markdown("##### ⚙️ 会话设置")
        cols = st.columns(2)
        with cols[0]:
            if st.button("➕ 开启新会话", use_container_width=True, disabled=is_generating):
                now = datetime.datetime.now()
                new_id = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
                st.session_state["conversation_id"] = new_id
                st.session_state["messages"] = load_history_from_db(new_id)
                st.session_state["vector_store"] = None
                st.session_state["local_tool"] = None
                st.session_state["rag_index_loaded_from_disk"] = False
                st.rerun()
        with cols[1]:
            if st.button("删除会话", use_container_width=True, disabled=is_generating):
                delete_conversation(cid)
                remaining = list_conversations_with_preview()
                if remaining:
                    st.session_state["conversation_id"] = remaining[0]["id"]
                    st.session_state["messages"] = load_history_from_db(remaining[0]["id"])
                else:
                    now = datetime.datetime.now()
                    new_id = create_conversation(f"新会话 {now:%Y-%m-%d %H:%M}")
                    st.session_state["conversation_id"] = new_id
                    st.session_state["messages"] = load_history_from_db(new_id)
                st.session_state["vector_store"] = None
                st.session_state["local_tool"] = None
                st.session_state["rag_index_loaded_from_disk"] = False
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ 清除所有历史记录", use_container_width=True, disabled=is_generating):
            clear_all_history()
            st.session_state.clear()
            st.rerun()


def visualize_tool_calls(intermediate_steps: List[Any]) -> None:
    """在界面上可视化工具调用过程与部分上下文内容。"""
    if not intermediate_steps:
        return

    with st.expander("🧠 Agent 工具调用与检索过程", expanded=False):
        for idx, (action, observation) in enumerate(intermediate_steps, start=1):
            tool_name = getattr(action, "tool", "unknown_tool")
            st.markdown(f"#### 步骤 {idx} · 调用工具：`{tool_name}`")
            st.markdown(f"**工具输入**：\n\n`{str(getattr(action, 'tool_input', ''))[:500]}`")

            # 针对不同工具类型，展示不同的“部分检索结果”
            if tool_name == "local_knowledge_search":
                st.markdown("**检索到的本地文档片段（节选）**：")
                try:
                    # Retriever 工具一般返回 Document 列表
                    docs = observation
                    if isinstance(docs, list):
                        for i, doc in enumerate(docs[:3], start=1):
                            content = ""
                            if hasattr(doc, "page_content"):
                                content = doc.page_content
                            elif isinstance(doc, dict):
                                content = doc.get("page_content", "")
                            snippet = (content or "").strip().replace("\n", " ")
                            if len(snippet) > 200:
                                snippet = snippet[:200] + "..."
                            st.markdown(f"- 片段 {i}: {snippet}")
                except Exception as e:
                    st.write(f"展示本地检索结果时出错：{e}")
            else:
                # 例如 Tavily 搜索结果：通常是 dict 列表，包含 url/title/score 等
                st.markdown("**工具返回（节选）**：")
                try:
                    if isinstance(observation, list):
                        for item in observation[:3]:
                            if isinstance(item, dict):
                                title = item.get("title") or "未提供标题"
                                url = item.get("url") or ""
                                snippet = item.get("content", "") or item.get("snippet", "")
                                if snippet:
                                    snippet = snippet.replace("\n", " ")
                                    if len(snippet) > 200:
                                        snippet = snippet[:200] + "..."
                                st.markdown(
                                    f"- **{title}**\n\n  链接：{url}\n\n  节选：{snippet}"
                                )
                    else:
                        st.code(str(observation)[:1000])
                except Exception as e:
                    st.write(f"展示工具返回结果时出错：{e}")


def extract_sources(intermediate_steps: List[Any]) -> List[Dict[str, str]]:
    """从工具调用结果中提取可展示的来源链接（主要用于 Tavily）。"""
    sources: List[Dict[str, str]] = []
    seen = set()
    for action, observation in intermediate_steps or []:
        tool_name = getattr(action, "tool", "")
        if tool_name not in {"tavily_search_results_json", "tavily_search"}:
            continue
        if isinstance(observation, list):
            for item in observation:
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or ""
                title = item.get("title") or url or "未提供标题"
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": title, "url": url})
    return sources


def typewriter_markdown(target, text: str) -> None:
    """将最终文本做“伪打字机效果”输出到 target（st.empty() 的占位符）。

    说明：
    - LangChain 的 AgentExecutor 往往不是逐 token 透传，因此这里采用“前端渐进渲染”。
    - 该方式无法减少等待时间，但能改善最终回答“瞬间弹出”的观感。
    """
    if not text:
        return
    n = len(text)
    # 文本越长，每次刷新显示的字符越多，避免太卡
    step = 6 if n < 800 else 12 if n < 2000 else 24
    delay = 0.008 if n < 800 else 0.005 if n < 2000 else 0.003
    buf = ""
    for i in range(0, n, step):
        buf = text[: i + step]
        target.markdown(buf)
        time.sleep(delay)
    target.markdown(text)


def main() -> None:
    """应用主函数。"""
    st.set_page_config(
        page_title="Advanced DeepSearch Agent",
        page_icon="🔍",
        layout="wide",
    )

    init_env()
    init_db()
    init_session_state()

    # 渲染侧边栏：会话控制 + 文档上传
    render_sidebar()

    # 主标题与介绍
    st.title("🔍 Advanced DeepSearch Agent")
    st.markdown(
        """
        一个面向商业场景的多引擎 AI Agent：同时具备**联网搜索**与**本地文档 RAG 检索**能力。
        """
    )

    # 聊天记录区域
    render_chat_history()

    # 欢迎页快捷提示词：仅当当前会话还没有任何用户提问时展示
    has_user_msg = any(m.get("role") == "user" for m in (st.session_state.get("messages") or []))
    if not has_user_msg:
        st.markdown("#### 🚀 快捷开始")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📄 总结这份文档", use_container_width=True, disabled=st.session_state.get("is_generating", False)):
                st.session_state["pending_user_input"] = "请总结这份文档/论文的核心内容，并用要点列出主要贡献与结论。"
                st.rerun()
        with c2:
            if st.button("🔥 今天微博热搜前三", use_container_width=True, disabled=st.session_state.get("is_generating", False)):
                st.session_state["pending_user_input"] = "请联网查询今天微博热搜前三是什么，并简要说明每条热搜的背景。"
                st.rerun()
        with c3:
            if st.button("🤖 介绍一下 DeepSeek", use_container_width=True, disabled=st.session_state.get("is_generating", False)):
                st.session_state["pending_user_input"] = "请介绍一下 DeepSeek 的产品与模型能力、典型应用场景，以及与同类模型相比的特点。"
                st.rerun()

    # 根据是否有本地文档，动态组装工具列表
    tools: List[Any] = []
    tavily_tool = None
    try:
        tavily_tool = get_tavily_tool()
        tools.append(tavily_tool)
    except Exception as e:
        st.error(f"初始化 Tavily 搜索工具失败：{e}")

    local_tool = st.session_state.get("local_tool")
    if local_tool is not None:
        tools.append(local_tool)

    if not tools:
        st.stop()

    try:
        agent_executor = build_agent(tools)
    except Exception as e:
        st.error(f"构建 Agent 失败：{e}")
        st.stop()

    # 用户输入区
    pending = (st.session_state.get("pending_user_input") or "").strip()
    user_input = st.chat_input(
        "请输入你的问题，例如：'总结这份文档的要点' 或 '最近的 AI 热点是什么？'",
        disabled=bool(st.session_state.get("is_generating", False)),
    )
    if not user_input and pending:
        user_input = pending
        st.session_state["pending_user_input"] = ""

    if not user_input:
        return

    # 记录用户消息（内存 + 数据库）
    cid = int(st.session_state["conversation_id"])
    st.session_state["messages"].append({"role": "user", "content": user_input})
    save_message_to_db(cid, "user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # 助手消息：流式输出 + 工具调用可视化
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        st.session_state["is_generating"] = True

        chat_history = convert_history_for_agent()

        final_state: Dict[str, Any] = {}
        full_answer = ""

        try:
            # 部分服务端（DeepSeek / 硅基流动 / Tavily）在高峰期会返回 503 busy。
            # 这里做一个轻量的重试（指数退避），提升“商业级”稳定性。
            max_attempts = 3
            base_sleep_s = 1.5
            last_error: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    for step in agent_executor.stream(
                        {
                            "input": user_input,
                            "chat_history": chat_history,
                        }
                    ):
                        final_state = step
                        chunk = step.get("output", "")
                        if isinstance(chunk, str) and chunk:
                            full_answer = chunk
                            answer_placeholder.markdown(full_answer)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    err_text = str(e)
                    is_busy = ("503" in err_text) or ("System is really busy" in err_text) or ("busy" in err_text.lower())
                    if attempt < max_attempts and is_busy:
                        sleep_s = base_sleep_s * (2 ** (attempt - 1))
                        answer_placeholder.info(
                            f"服务端繁忙，正在自动重试（第 {attempt}/{max_attempts} 次失败，{sleep_s:.1f}s 后重试）..."
                        )
                        time.sleep(sleep_s)
                        continue
                    raise

            if last_error is not None:
                raise last_error
        except Exception as e:
            st.error(
                "Agent 调用过程中出现错误。"
                "如果是 503/繁忙类错误，通常是服务端高峰期限流，稍后重试即可。\n\n"
                f"详细信息：{e}"
            )
            return
        finally:
            st.session_state["is_generating"] = False

        # 可视化工具调用过程与部分上下文
        intermediate_steps: List[Any] = final_state.get("intermediate_steps", [])
        sources = extract_sources(intermediate_steps)

        # 最终回答：用“伪打字机”效果再渲染一次（改善观感）
        # 注意：此处会覆盖上面流式阶段最后一次渲染的内容
        typewriter_markdown(answer_placeholder, full_answer)

        # 将“来源/链接/工具调用过程”折叠起来展示（默认收起）
        if sources or intermediate_steps:
            with st.expander("🧾 来源链接 & 工具检索过程（点击展开）", expanded=False):
                if sources:
                    st.markdown("#### Sources（联网检索来源）")
                    for s in sources:
                        st.markdown(f"- [{s['title']}]({s['url']})")
                if intermediate_steps:
                    st.markdown("---")
                    visualize_tool_calls(intermediate_steps)

        # 最终回答落盘
        # 注意：上面已渲染 full_answer，这里不再重复渲染

    st.session_state["messages"].append({"role": "assistant", "content": full_answer})
    save_message_to_db(cid, "assistant", full_answer)


if __name__ == "__main__":
    main()

