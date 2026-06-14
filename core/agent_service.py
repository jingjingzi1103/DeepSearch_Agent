"""Agent 业务层：与 UI 无关的 LLM / RAG / 工具组装 / 对话执行逻辑。

Phase C Step 1：从 advanced_agent.py 抽离，供 Streamlit 与后续 FastAPI 共用。
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from core.env_loader import get_env_key, load_project_env
from core.evaluator import evaluate_answer, extract_context_from_steps
from core.history_utils import convert_messages_for_agent
from core.intent_utils import is_document_rag_query, is_realtime_query, is_weibo_hot_query
from core.platform_guard import build_platform_disclaimer, get_preflight_notice
from core.rag_pipeline import run_rag_pipeline
from core.realtime import get_enabled_realtime_tools
from core.realtime_pipeline import agent_used_tool, run_weibo_hot_pipeline
from core.storage import get_faiss_dir_for_conversation
from core.trace_utils import build_trace_metadata


###########################################################################
# 环境与健康检查
###########################################################################


def init_env() -> None:
    """加载 .env 中的环境变量。"""
    load_project_env()

    tracing_v2 = (os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    langchain_api_key = (os.getenv("LANGCHAIN_API_KEY") or "").strip()
    enabled = tracing_v2 in {"1", "true", "yes", "on"} and bool(langchain_api_key)
    print(f"[LangSmith] tracing_v2 enabled: {enabled}")
    if enabled:
        print(
            "[LangSmith] 若终端报 403 Forbidden，请把 .env 中 LANGCHAIN_TRACING_V2 改为 false"
            "（不影响对话与 Judge）。"
        )


def probe_deepseek_key() -> Optional[str]:
    """启动时探测 DeepSeek Key；失败则返回可读错误信息。"""
    api_key = get_env_key("DEEPSEEK_API_KEY")
    if not api_key.startswith("sk-"):
        return "DEEPSEEK_API_KEY 格式异常（应以 sk- 开头），请检查 .env 是否复制完整。"

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
        ).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20):
            return None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 401:
            return (
                "DeepSeek API Key 无效（401）。请到 https://platform.deepseek.com 确认：\n"
                "1) Key 是否已删除/过期；2) 是否复制完整（无多余空格）；3) 账户是否有余额。\n"
                f"原始响应：{body[:200]}"
            )
        return f"DeepSeek 连接异常（HTTP {e.code}）：{body[:200]}"
    except Exception as e:
        return f"DeepSeek 连接失败：{e}"


###########################################################################
# LLM / Embedding
###########################################################################


def get_llm() -> ChatOpenAI:
    """创建 DeepSeek Chat LLM 客户端。"""
    api_key = get_env_key("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.3,
    )


def get_embedding_model() -> OpenAIEmbeddings:
    """创建硅基流动 BGE-M3 Embedding 客户端。"""
    api_key = get_env_key("SILICONFLOW_API_KEY")
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        model="BAAI/bge-m3",
        chunk_size=64,
    )


###########################################################################
# RAG：文档向量化与检索工具
###########################################################################


@dataclass
class VectorStoreBuildResult:
    vector_store: Optional[FAISS] = None
    error: Optional[str] = None
    warning: Optional[str] = None


def build_vector_store_from_file(uploaded_file, conversation_id: int) -> VectorStoreBuildResult:
    """根据上传文件构建 FAISS 向量库（与 UI 框架无关）。"""
    if uploaded_file is None:
        return VectorStoreBuildResult(error="未提供上传文件")

    file_name = uploaded_file.name.lower()
    raw_text = ""

    try:
        if file_name.endswith(".pdf"):
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
            return VectorStoreBuildResult(error="目前仅支持 PDF 和 TXT 文件。")
    except Exception as e:
        return VectorStoreBuildResult(error=f"解析文档时出错：{e}")

    if not raw_text.strip():
        return VectorStoreBuildResult(warning="文档中未提取到有效文本内容。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )
    docs = splitter.create_documents([raw_text])

    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(docs, embeddings)
        index_dir = get_faiss_dir_for_conversation(conversation_id)
        os.makedirs(index_dir, exist_ok=True)
        vector_store.save_local(index_dir)
        return VectorStoreBuildResult(vector_store=vector_store)
    except Exception as e:
        return VectorStoreBuildResult(error=f"构建向量库失败：{e}")


def build_retriever_tool(vector_store: FAISS):
    """将向量库包装为 LangChain Retriever 工具。"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return create_retriever_tool(
        retriever=retriever,
        name="local_knowledge_search",
        description=(
            "基于用户上传的本地文档进行语义检索，"
            "适合回答与该文档内容紧密相关的问题，例如“根据文档内容总结第3章”之类。"
        ),
    )


def load_vector_store_for_conversation(conversation_id: int) -> VectorStoreBuildResult:
    """从磁盘加载本会话已落盘的 FAISS 向量库。"""
    index_dir = get_faiss_dir_for_conversation(conversation_id)
    if not os.path.isdir(index_dir):
        return VectorStoreBuildResult()
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return VectorStoreBuildResult(vector_store=vector_store)
    except Exception as e:
        return VectorStoreBuildResult(error=f"加载向量库失败：{e}")


###########################################################################
# 工具组装与 Agent 构建
###########################################################################


def get_tavily_tool() -> TavilySearchResults:
    """创建 Tavily 联网搜索工具。"""
    get_env_key("TAVILY_API_KEY")
    return TavilySearchResults(
        max_results=8,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )


def assemble_agent_tools(
    *,
    enable_weibo_hot: bool = True,
    enable_zhipu_search: bool = True,
    local_tool: Any = None,
) -> List[Any]:
    """按开关动态组装 Agent 可用工具列表。"""
    tools: List[Any] = []
    realtime_tools = get_enabled_realtime_tools()
    if not enable_weibo_hot:
        realtime_tools = [t for t in realtime_tools if getattr(t, "name", "") != "weibo_hot_search"]
    if not enable_zhipu_search:
        realtime_tools = [t for t in realtime_tools if getattr(t, "name", "") != "zhipu_web_search"]
    tools.extend(realtime_tools)
    tools.append(get_tavily_tool())
    if local_tool is not None:
        tools.append(local_tool)
    return tools


def build_agent(tools: List[Any]) -> AgentExecutor:
    """根据给定工具列表构建多工具 AgentExecutor。"""
    llm = get_llm()

    now = datetime.datetime.now()
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    weekday_cn = weekday_map[now.weekday()]
    current_time_str = f"今天是 {now:%Y-%m-%d}（星期{weekday_cn}），当前时间 {now:%H:%M:%S}。"

    has_local_tool = any(getattr(t, "name", "") == "local_knowledge_search" for t in tools)
    has_weibo_tool = any(getattr(t, "name", "") == "weibo_hot_search" for t in tools)
    has_zhipu_tool = any(getattr(t, "name", "") == "zhipu_web_search" for t in tools)

    local_tool_hard_rule = ""
    weibo_tool_hard_rule = ""
    zhipu_tool_hard_rule = ""
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
    if has_weibo_tool:
        weibo_tool_hard_rule = (
            "\n【微博热搜硬规则（必须遵守）】\n"
            "当用户询问微博热搜、热搜前三/前十、微博热点排行时：\n"
            "1) 必须优先调用 `weibo_hot_search` 获取实时榜单，不得凭记忆或 Tavily 旧闻回答。\n"
            "2) 回答中按排名列出词条，并注明数据抓取时间（工具返回的 fetched_at）。\n"
            "3) 若 `weibo_hot_search` 失败，再 fallback 到 Tavily / zhipu_web_search，并说明实时接口不可用。\n"
        )
    if has_zhipu_tool:
        zhipu_tool_hard_rule = (
            "\n【智谱联网搜索硬规则（必须遵守）】\n"
            "当用户询问中文新闻、国内热点、行业动态、人物/公司产品（非微博热搜榜）时：\n"
            "1) 必须优先调用 `zhipu_web_search` 获取检索结果，再基于返回内容回答。\n"
            "2) 回答需引用来源标题与链接，并标注 publish_date（若有）。\n"
            "3) 不得在未调用工具的情况下编造新闻或数据。\n"
            "4) 【跨平台禁止推断】若用户问 A 平台（如抖音）热榜/是否上热门，"
            "   但工具返回中没有 A 平台的直接依据，必须明确写「无法从检索结果确认」，"
            "   只可陈述其他平台有来源依据的内容，并标注出处；禁止用微博/新闻推断抖音热榜。\n"
        )

    realtime_rule = (
        "\n【实时信息检索硬规则（必须遵守）】\n"
        "当用户询问「今天/此刻/最近/热搜/热点/新闻」等时效性问题时：\n"
        "1) 必须先调用联网工具（微博榜用 weibo_hot_search；其他优先 zhipu_web_search，必要时 Tavily），"
        f"   搜索词应包含今天的日期（{now:%Y-%m-%d}）。\n"
        "2) 优先采用来源中明确标注为今天或最近几小时的内容；"
        "   若只有旧闻、无法确认是否为当前时刻榜单，必须如实说明「未能获取到当前时刻的准确排名」。\n"
        "3) 不得用模型记忆里的旧热搜冒充「今天的热搜」。\n"
        "4) 回答需附来源链接；若能从来源中看到发布/更新时间，一并标注。\n"
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
                    f"{realtime_rule}"
                    f"{weibo_tool_hard_rule}"
                    f"{zhipu_tool_hard_rule}"
                    f"{local_tool_hard_rule}"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


###########################################################################
# 对话执行（供 Streamlit / FastAPI 共用）
###########################################################################


def prepare_chat_history(
    messages: List[Dict[str, Any]],
    user_input: str,
    *,
    local_rag_enabled: bool,
) -> List[Dict[str, str]]:
    """将历史消息转换为 LangChain 可用格式（不依赖 session_state）。"""
    history = list(messages or [])
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    strip_ai = is_realtime_query(user_input) or is_document_rag_query(
        user_input, local_rag_enabled=local_rag_enabled
    )
    return convert_messages_for_agent(
        history,
        max_messages=10,
        strip_assistant_for_realtime=strip_ai,
    )


@dataclass
class ChatTurnResult:
    full_answer: str = ""
    intermediate_steps: List[Any] = field(default_factory=list)
    trace_metadata: Dict[str, Any] = field(default_factory=dict)
    eval_result: Optional[Dict[str, Any]] = None
    preflight_notice: Optional[str] = None
    used_forced_pipeline: bool = False
    error: Optional[str] = None


def run_chat_turn(
    *,
    user_input: str,
    agent_executor: AgentExecutor,
    messages: List[Dict[str, Any]],
    vector_store: Any,
    local_rag_enabled: bool,
    enable_weibo_hot: bool = True,
    enable_forced_rag: bool = True,
    enable_judge: bool = True,
    judge_provider: Optional[str] = None,
    enabled_tool_names: Optional[List[str]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_answer_chunk: Optional[Callable[[str], None]] = None,
) -> ChatTurnResult:
    """执行一轮对话：强制链路 / Agent 流式 / 兜底 / Judge。"""
    result = ChatTurnResult()
    result.preflight_notice = get_preflight_notice(
        user_input, enabled_tool_names or []
    )

    chat_history = prepare_chat_history(
        messages, user_input, local_rag_enabled=local_rag_enabled
    )

    use_forced_weibo = is_weibo_hot_query(user_input) and enable_weibo_hot
    use_forced_rag = (
        enable_forced_rag
        and is_document_rag_query(user_input, local_rag_enabled=local_rag_enabled)
        and not use_forced_weibo
    )

    def _emit_status(msg: str) -> None:
        if on_status:
            on_status(msg)

    def _emit_answer(text: str) -> None:
        if on_answer_chunk:
            on_answer_chunk(text)

    try:
        if use_forced_weibo:
            result.used_forced_pipeline = True
            _emit_status("检测到微博热搜问题，正在强制拉取实时接口（不依赖 Agent 自行调工具）...")
            llm = get_llm()
            answer, steps, trace = run_weibo_hot_pipeline(user_input, llm)
            result.full_answer = answer
            result.intermediate_steps = steps
            result.trace_metadata = trace
            _emit_answer(answer)
        elif use_forced_rag:
            result.used_forced_pipeline = True
            _emit_status("检测到文档相关问题，正在强制检索本地向量库（不依赖 Agent 自行调工具）...")
            llm = get_llm()
            answer, steps, trace = run_rag_pipeline(user_input, llm, vector_store)
            result.full_answer = answer
            result.intermediate_steps = steps
            result.trace_metadata = trace
            _emit_answer(answer)
        else:
            full_answer = ""
            final_state: Dict[str, Any] = {}
            max_attempts = 3
            base_sleep_s = 1.5
            last_error: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    for step in agent_executor.stream(
                        {"input": user_input, "chat_history": chat_history}
                    ):
                        final_state = step
                        chunk = step.get("output", "")
                        if isinstance(chunk, str) and chunk:
                            full_answer = chunk
                            _emit_answer(full_answer)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    err_text = str(e)
                    is_busy = (
                        ("503" in err_text)
                        or ("System is really busy" in err_text)
                        or ("busy" in err_text.lower())
                    )
                    if attempt < max_attempts and is_busy:
                        sleep_s = base_sleep_s * (2 ** (attempt - 1))
                        _emit_status(
                            f"服务端繁忙，正在自动重试（第 {attempt}/{max_attempts} 次失败，"
                            f"{sleep_s:.1f}s 后重试）..."
                        )
                        time.sleep(sleep_s)
                        continue
                    raise

            if last_error is not None:
                raise last_error

            result.full_answer = full_answer
            result.intermediate_steps = final_state.get("intermediate_steps", []) or []

            need_force_weibo = (
                is_weibo_hot_query(user_input)
                and enable_weibo_hot
                and not agent_used_tool(result.intermediate_steps, "weibo_hot_search")
            )
            if need_force_weibo:
                _emit_status("Agent 未调用微博实时接口，已自动切换为强制拉取模式...")
                llm = get_llm()
                answer, steps, trace = run_weibo_hot_pipeline(user_input, llm)
                result.full_answer = answer
                result.intermediate_steps = steps
                result.trace_metadata = trace
                result.used_forced_pipeline = True
                _emit_answer(answer)
            else:
                need_force_rag = (
                    enable_forced_rag
                    and is_document_rag_query(user_input, local_rag_enabled=local_rag_enabled)
                    and not agent_used_tool(result.intermediate_steps, "local_knowledge_search")
                )
                if need_force_rag:
                    _emit_status("Agent 未调用本地文档检索，已自动切换为强制 RAG 模式...")
                    llm = get_llm()
                    answer, steps, trace = run_rag_pipeline(user_input, llm, vector_store)
                    result.full_answer = answer
                    result.intermediate_steps = steps
                    result.trace_metadata = trace
                    result.used_forced_pipeline = True
                    _emit_answer(answer)
                else:
                    result.trace_metadata = build_trace_metadata(result.intermediate_steps)

    except Exception as e:
        result.error = (
            "Agent 调用过程中出现错误。"
            "如果是 503/繁忙类错误，通常是服务端高峰期限流，稍后重试即可。\n\n"
            f"详细信息：{e}"
        )
        return result

    if not result.trace_metadata:
        result.trace_metadata = build_trace_metadata(result.intermediate_steps)

    disclaimer = build_platform_disclaimer(user_input, result.trace_metadata)
    if disclaimer:
        result.full_answer = f"{result.full_answer.rstrip()}\n\n---\n\n{disclaimer}"

    if enable_judge and result.full_answer.strip():
        ctx = extract_context_from_steps(result.intermediate_steps)
        result.eval_result = evaluate_answer(
            user_input, result.full_answer, ctx, judge_provider=judge_provider
        )

    return result
