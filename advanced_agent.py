"""兼容旧入口 → 转发到薄 SSE 客户端。

推荐启动方式（需先启动 API）：
    uvicorn api.main:app --reload --port 8000
    streamlit run client/streamlit_app.py

本文件保留以便旧文档/习惯仍可使用：
    streamlit run advanced_agent.py
"""

from client.streamlit_app import main

if __name__ == "__main__":
    main()
