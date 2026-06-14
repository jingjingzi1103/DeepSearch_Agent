"""统一加载 .env，避免系统环境变量覆盖或 Key 带空格导致 401。"""

import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

_KEY_NAMES = (
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
    "SILICONFLOW_API_KEY",
    "LANGCHAIN_API_KEY",
    "ZHIPU_API_KEY",
)


def load_project_env() -> None:
    """从项目根目录加载 .env，.env 优先于系统里已有的同名变量。"""
    load_dotenv(ENV_PATH, override=True)
    for name in _KEY_NAMES:
        raw = os.getenv(name)
        if raw is not None:
            os.environ[name] = raw.strip().strip('"').strip("'")


def get_env_key(name: str, *, required: bool = True) -> str:
    load_project_env()
    value = (os.getenv(name) or "").strip()
    if required and not value:
        raise ValueError(f"未检测到 {name}，请在 {ENV_PATH} 中配置。")
    return value
