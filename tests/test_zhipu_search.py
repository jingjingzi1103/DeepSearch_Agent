"""智谱 Web Search 解析测试（不调用外部 API）。"""

from unittest.mock import MagicMock, patch

from core.realtime.zhipu_search import fetch_zhipu_web_search
from core.trace_utils import extract_sources_from_steps, parse_json_tool_items


def test_parse_zhipu_tool_json():
    raw = (
        '{"provider":"zhipu_web_search","query":"AI","items":['
        '{"rank":1,"title":"新闻A","url":"https://example.com/a","content":"摘要"}'
        "]}"
    )
    items = parse_json_tool_items(raw)
    assert len(items) == 1
    sources = extract_sources_from_steps(
        [{"tool": "zhipu_web_search", "tool_input": {"query": "AI"}, "observation": raw}]
    )
    assert sources[0]["source_type"] == "zhipu"
    assert sources[0]["url"] == "https://example.com/a"


@patch("core.realtime.zhipu_search.urllib.request.urlopen")
@patch("core.realtime.zhipu_search.get_env_key", return_value="test-key")
def test_fetch_zhipu_web_search_parses_response(mock_key, mock_urlopen):
    payload = {
        "search_result": [
            {
                "title": "测试标题",
                "link": "https://news.test/1",
                "content": "正文",
                "publish_date": "2025-06-01",
            }
        ]
    }
    resp = MagicMock()
    resp.read.return_value = __import__("json").dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    mock_urlopen.return_value = resp

    items = fetch_zhipu_web_search("测试", count=3)
    assert len(items) == 1
    assert items[0]["title"] == "测试标题"
    assert items[0]["url"] == "https://news.test/1"
