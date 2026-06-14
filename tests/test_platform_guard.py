"""平台能力边界检测测试。"""

from core.platform_guard import (
    build_platform_disclaimer,
    get_preflight_notice,
    is_douyin_hot_query,
)


def test_is_douyin_hot_query():
    assert is_douyin_hot_query("张子墨最近抖音有没有上热榜")
    assert not is_douyin_hot_query("介绍一下 DeepSeek")


def test_preflight_notice_for_douyin():
    msg = get_preflight_notice("抖音热榜", ["zhipu_web_search", "tavily_search"])
    assert msg is not None
    assert "抖音" in msg


def test_disclaimer_when_no_douyin_evidence():
    trace = {
        "steps": [
            {
                "tool": "zhipu_web_search",
                "tool_input": {"query": "张子墨"},
                "observation": (
                    '{"items":[{"title":"某新闻","url":"https://news.com/a",'
                    '"content":"关于微博活动的报道"}]}'
                ),
            }
        ]
    }
    disclaimer = build_platform_disclaimer("张子墨抖音有没有上热榜", trace)
    assert "未接入抖音" in disclaimer or "无法确认" in disclaimer
