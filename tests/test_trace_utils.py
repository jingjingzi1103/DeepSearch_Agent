"""trace 序列化与来源提取测试。"""

from core.trace_utils import (
    build_trace_metadata,
    extract_sources_from_steps,
    loads_trace_metadata,
    dumps_trace_metadata,
    serialize_intermediate_steps,
)


class _FakeAction:
    def __init__(self, tool: str, tool_input: str = ""):
        self.tool = tool
        self.tool_input = tool_input


def test_serialize_tavily_steps():
    steps = [
        (
            _FakeAction("tavily_search_results_json", "AI news"),
            [{"title": "新闻", "url": "https://example.com/a", "content": "摘要"}],
        )
    ]
    serialized = serialize_intermediate_steps(steps)
    assert serialized[0]["tool"] == "tavily_search_results_json"
    meta = build_trace_metadata(steps)
    assert len(meta["sources"]) == 1
    assert meta["sources"][0]["url"] == "https://example.com/a"


def test_extract_weibo_sources_from_json_string():
    obs = (
        '{"provider":"weibo_hot_search","items":['
        '{"rank":1,"title":"话题A","url":"https://s.weibo.com/weibo?q=a","fetched_at":"2025-06-05"}'
        "]}"
    )
    steps = [{"tool": "weibo_hot_search", "tool_input": {"limit": 3}, "observation": obs}]
    sources = extract_sources_from_steps(steps)
    assert len(sources) == 1
    assert sources[0]["source_type"] == "weibo"


def test_metadata_json_roundtrip():
    meta = {"sources": [{"title": "x", "url": "https://x.com"}], "steps": []}
    raw = dumps_trace_metadata(meta)
    loaded = loads_trace_metadata(raw)
    assert loaded["sources"][0]["title"] == "x"
