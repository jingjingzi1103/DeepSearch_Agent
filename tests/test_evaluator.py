"""Judge 模块单元测试（不调用外部 API）。"""

import os
from unittest.mock import patch

import pytest

from core.evaluator import (
    extract_context_from_steps,
    list_available_judge_providers,
    resolve_judge_config,
    _parse_judge_json,
)


class _FakeAction:
    def __init__(self, tool: str):
        self.tool = tool


def test_parse_judge_json_from_codeblock():
    raw = '```json\n{"faithfulness":4,"relevance":5,"has_citation":true,"overall":4.5,"reason":"ok"}\n```'
    result = _parse_judge_json(
        raw, judge_model="智谱 GLM / glm-4-flash", judge_provider="zhipu"
    )
    assert result["faithfulness"] == 4.0
    assert result["relevance"] == 5.0
    assert result["has_citation"] is True
    assert result["overall"] == 4.5
    assert result["reason"] == "ok"
    assert result["judge_provider"] == "zhipu"


def test_resolve_judge_config_deepseek():
    with patch.dict(os.environ, {"JUDGE_PROVIDER": "deepseek"}, clear=False):
        cfg = resolve_judge_config()
    assert cfg["provider"] == "deepseek"
    assert "deepseek" in cfg["model"]


def test_resolve_judge_config_zhipu_override():
    cfg = resolve_judge_config("zhipu")
    assert cfg["provider"] == "zhipu"
    assert cfg["model"] == "glm-4-flash"


def test_resolve_judge_config_invalid():
    with pytest.raises(ValueError, match="不支持"):
        resolve_judge_config("openai")


def test_list_available_judge_providers(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-test")
    providers = list_available_judge_providers()
    names = {p["provider"] for p in providers}
    assert "deepseek" in names
    assert "zhipu" in names


def test_extract_context_from_tavily_steps():
    steps = [
        (
            _FakeAction("tavily_search_results_json"),
            [{"title": "新闻A", "url": "https://a.com", "content": "摘要内容"}],
        )
    ]
    ctx = extract_context_from_steps(steps)
    assert "tavily_search_results_json" in ctx
    assert "新闻A" in ctx
    assert "https://a.com" in ctx


def test_extract_context_empty():
    assert "未调用工具" in extract_context_from_steps([])
